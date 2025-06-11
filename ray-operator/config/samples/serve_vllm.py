import yaml
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
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser

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
            if not self.openai_serving_chat:
                model_config = await self.engine.get_model_config()
                served_model_names = [engine_args.model]
                self.openai_serving_chat = OpenAIServingChat(
                    self.engine, model_config, served_model_names, "assistant"
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

# Entrypoint for Ray Serve
app = build_app("vllm_config.yaml")

try:
    config_path = os.environ.get('VLLM_CONFIG', '/home/ray/vllm_config.yaml')
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        model = build_app(config)
except FileNotFoundError:
    logger.error(f"Configuration file not found at {config_path}")
    sys.exit(1)
except json.JSONDecodeError:
    logger.error(f"Invalid JSON in configuration file {config_path}")
    sys.exit(1)