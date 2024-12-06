from typing import Optional
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import HookedSAETransformer, SAE, SAEConfig 
from typing import Optional, List, Tuple
from fastapi import FastAPI


# Initialize the FastAPI app
app = FastAPI()

# Get the SAEs
saes_dict = {}


for l in range(5,6):

    sae, cfg_dict, sparsity = SAE.from_pretrained(
            release = "gpt2-small-res-jb",
            sae_id = f"blocks.{l}.hook_resid_pre",
            device = "cpu"
            )
    saes_dict[f"blocks.{l}.hook_resid_pre"] = sae
    if l == 0:
        cfg = cfg_dict

sae = saes_dict["blocks.5.hook_resid_pre"]




# Load the model and tokenizer
model_name = "gpt2"  # You can choose any other model

model = HookedSAETransformer.from_pretrained(model_name)
tokenizer = model.tokenizer

# Ensure the model is in evaluation mode
model.eval()

# Define the request body model
class RequestBody(BaseModel):
    prompt: str
    max_length: int = 50  # Default max length of generated text
    return_activations: bool = False  # Flag to return activations

# Define the response model
class ResponseModel(BaseModel):
    generated_text: str
    activations: List = None  # Optional field for activations


def join_cache(cache: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(cache, dim=1)

def generate_and_cache(model: HookedSAETransformer,
                       prompt: str,
                       max_new_tokens: int,
                       sae: Optional[SAE | List[SAE]],
                       **gen_kwargs: Optional[dict],
                       ) -> Tuple[torch.Tensor, List[dict]]:

    cache = []
    if isinstance(sae, list):
        prepend_bos = sae[0].cfg.prepend_bos
    elif isinstance(sae, SAE): 
        prepend_bos = sae.cfg.prepend_bos
    else:
        prepend_bos = True
        


    def caching_hook(activations, hook):
        cache.append(activations)

    input_ids = model.to_tokens(prompt, prepend_bos=prepend_bos)
    if sae:

        with model.hooks(fwd_hooks=[(sae.cfg.hook_name, caching_hook)]):
            output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    **gen_kwargs
                    )
            cache = join_cache(cache)
            return output, cache
    else: 
        output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                **gen_kwargs
                )
        return output, []



@app.post("/generate", response_model=ResponseModel)
async def generate_response(request_body: RequestBody):
    # Tokenize the input prompt

    # Generate text
    with torch.no_grad():
        output, cache = generate_and_cache(model,request_body.prompt,max_new_tokens=50,sae=sae)  

    # Decode the generated text
    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True) 
    generated_text = generated_text[0]
    activations = None
    if request_body.return_activations:
        activations = cache.tolist() # Convert activations to list (requrired for pydantic serialization)

    return ResponseModel(generated_text=generated_text, activations=activations)

# To run the app, use the command:
# uvicorn server:app 
# curl -X POST "http://127.0.0.1:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "Once upon a time in a world far away, there lived a", "max_length": 100, "return_activations": true}'




