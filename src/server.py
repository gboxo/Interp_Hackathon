from typing import Optional, Any, List, Tuple
from pydantic import BaseModel
import numpy as np
import os
import torch
from transformers import AutoTokenizer
from sae_lens import HookedSAETransformer, SAE
from fastapi import FastAPI
import argparse
import uvicorn
import asyncio
from dataclasses import dataclass
import threading

# Constants
HOST = "127.0.0.1"
PORT = 8000

# Initialize the FastAPI app
app = FastAPI()
shutdown_event = asyncio.Event()
ready_event = threading.Event()

@dataclass
class BaseArgs:
    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        for key, value in vars(self).items():
            parser.add_argument(f"--{key}", type=type(value), default=None)
        return parser.parse_args()

    def __post_init__(self) -> None:
        command_line_args = self.parse_args()
        extra_args = set(vars(command_line_args)) - set(vars(self))
        if extra_args:
            raise ValueError(f"Unknown argument: {extra_args}")
        self.update(command_line_args)

    def update(self, args: Any) -> None:
        for key, value in vars(args).items():
            if value is not None:
                print(f"From command line, setting {key} to {value}")
                setattr(self, key, value)

@dataclass
class ServerConfig(BaseArgs):
    lm_model_name: str = "gpt2"
    sae_name: str = ""
    sae_release: str = "gpt2-small-res-jb"
    sae_layer: int = 5
    sae_locations: str = "hook_resid_pre"
    n_samples: int = 100
    max_length: int = 1024
    shuffle_options: bool = True
    dataset_name: str = "cais/mmlu"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_folder: str = "output"
    h5_output: str = "output_data.h5"
    reduce_activations: bool = True
    seed: int = 123

def load_sae(layer: int = 5, location: str = "hook_resid_pre", release: str = "gpt2-small-res-jb", device: str = "cpu") -> SAE:
    sae, _, _ = SAE.from_pretrained(
        release=release,
        sae_id=f"blocks.{layer}.{location}",
        device=device
    )
    return sae

# Define the request body model
class RequestBody(BaseModel):
    prompt: str
    max_length: int
    return_activations: bool

# Define the response model
class ResponseModel(BaseModel):
    generated_text: str
    activations: Optional[Any] = None

model: Optional[HookedSAETransformer] = None
sae: Optional[SAE] = None

@app.on_event("startup")
async def startup():
    global model, tokenizer, sae
    args = server_config
    model = HookedSAETransformer.from_pretrained(args.lm_model_name)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_name)
    sae = load_sae(layer=args.sae_layer, location=args.sae_locations, release=args.sae_release, device=args.device)
    model.add_sae(sae)
    ready_event.set()

@app.post("/shutdown")
async def shutdown():
    print("Shutting down the server...")
    shutdown_event.set()
    return {"message": "Shutting down the server..."}

async def run():
    config = uvicorn.Config(app=app, host=HOST, port=PORT)
    server = uvicorn.Server(config)

    # Start the server
    await server.serve()

async def main():
    try:
        await run()
    except Exception as e:
        print(f"An error occurred: {e}")

def join_cache(cache: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(cache, dim=1)

def generate_and_cache(model: HookedSAETransformer, prompt: str, max_new_tokens: int, sae: Optional[SAE | List[SAE]], **gen_kwargs: Optional[dict]) -> Tuple[torch.Tensor, Any]:
    prepend_bos = True
    if isinstance(sae, list):
        prepend_bos = sae[0].cfg.prepend_bos
    elif isinstance(sae, SAE):
        prepend_bos = sae.cfg.prepend_bos

    cache = []
    
    def caching_hook(activations, hook):
        cache.append(activations)

    input_ids = model.to_tokens(prompt, prepend_bos=prepend_bos)
    if sae:
        with model.hooks(fwd_hooks=[(sae.cfg.hook_name+".hook_sae_acts_post", caching_hook)]):
            output = model.generate(input_ids, max_new_tokens=max_new_tokens,verbose=False, **gen_kwargs)
            cache = join_cache(cache)
            cache = cache.detach().to("cpu").numpy().tolist()
            return output, cache
    else:
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, **gen_kwargs)
        return output, []

@app.post("/generate", response_model=ResponseModel)
async def generate_response(request_body: RequestBody):
    with torch.no_grad():
        output, cache = generate_and_cache(model, request_body.prompt, max_new_tokens=request_body.max_length, sae=sae)

    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    activations = cache if request_body.return_activations else None

    return ResponseModel(generated_text=generated_text, activations=activations)

def run_server(args):
    global server_config
    server_config = ServerConfig(
        lm_model_name=args.lm_model_name,
        sae_name=args.sae_name,
        sae_release=args.sae_release,
        sae_layer=args.sae_layer,
        sae_locations=args.sae_locations,
        n_samples=args.n_samples,
        max_length=args.max_length,
        shuffle_options=args.shuffle_options,
        dataset_name=args.dataset_name,
        output_folder=args.output_folder,
        h5_output=args.h5_output,
        reduce_activations=args.reduce_activations,
        seed=args.seed
    )

    uvicorn.run(app, host=HOST, port=PORT)

if __name__ == "__main__":
    asyncio.run(main())

