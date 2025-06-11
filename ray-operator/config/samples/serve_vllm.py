import yaml
import os
import logging
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser
#from vllm.entrypoints.logger import RequestLogger

# Helper to parse YAML config into vLLM CLI-style arguments
def parse_vllm_args(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Convert YAML to CLI-style args for vLLM
    args = []
    for k, v in config.items():
        args.append(f"--{k.replace('_', '-')}")
        args.append(str(v))
    return args

# Build Ray Serve deployment
def build_app(config_path="vllm_config.yaml"):
    cli_args = parse_vllm_args(config_path)
    # Parse vLLM engine args
    arg_parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(arg_parser)
    parsed_args = parser.parse_args(args=cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)

    app = FastAPI()
    logger = logging.getLogger("ray.serve")

    @serve.deployment
    @serve.ingress(app)
    class VLLMDeployment:
        def __init__(self):
            logger.info(f"Starting vLLM engine with args: {engine_args}")
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.openai_serving_chat = None

        @app.post("/v1/chat/completions")
        async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
            if engine_args.served_model_name is not None:
                served_model_names = engine_args.served_model_name
            else:
                served_model_names = [engine_args.model]

            # if engine_args.max_log_len:
            #     request_logger = RequestLogger(max_log_len=engine_args.max_log_len)
            # else:
            #     request_logger = None

            if not self.openai_serving_chat:
                model_config = await self.engine.get_model_config()
                base_model_paths = [
                    BaseModelPath(name=name, model_path=engine_args.model)
                    for name in served_model_names
                ]
                openai_serving_models = OpenAIServingModels(
                    engine_client=self.engine,
                    model_config=model_config,
                    base_model_paths=base_model_paths,
                    lora_modules=None,
                    prompt_adapters=None,
                )
                self.openai_serving_chat = OpenAIServingChat(
                    self.engine,
                    model_config,
                    openai_serving_models,
                    "assistant",
                    request_logger=None,
                    chat_template=None,
                    chat_template_content_format="auto",
                )
            generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)
            if isinstance(generator, ErrorResponse):
                return JSONResponse(content=generator.model_dump(), status_code=generator.code)
            if request.stream:
                return StreamingResponse(content=generator, media_type="text/event-stream")
            else:
                assert isinstance(generator, ChatCompletionResponse)
                return JSONResponse(content=generator.model_dump())

    return VLLMDeployment.bind()


config_path = os.environ.get('VLLM_CONFIG', '/home/ray/vllm_config.yaml')
model = build_app(config_path)
