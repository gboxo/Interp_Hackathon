"""
Join the prompts and responses into a single body of text.
Tokenize the text and upload it to Hugging Face's Datasets.
"""

import os
import json
import datasets
from datasets import Dataset
from transformer_lens import HookedTransformer
import torch
# Load the model and tokenizer
model = HookedTransformer.from_pretrained("gemma-2-2b-it")
tokenizer = model.tokenizer
del model

def load_prompts(file_path):
    """Load prompts from a JSON file."""
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def load_arguments(file_path):
    """Load arguments from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def prepare_data(prompts, arguments):
    """Combine prompts and arguments into a single dataset."""
    full_dict = []
    for arg, prompt in zip(arguments.values(), prompts):
        combined_prompt = f"{prompt['prompt']}\n{arg}"
        tokens = tokenizer(combined_prompt, return_tensors="pt")

        prompt["tokens"] = tokens["input_ids"].squeeze().tolist()
        prompt["prompt"] = combined_prompt  # Update prompt with combined text
        full_dict.append(prompt)
    return full_dict


def create_dataset(full_dict):
    """Create a Hugging Face dataset from full_dict."""
    return Dataset.from_dict({key: [d[key] for d in full_dict] for key in full_dict[0]})

def upload_to_huggingface(dataset, dataset_name):
    """Upload the dataset to Hugging Face."""
    dataset.push_to_hub(dataset_name)

# Load data
prompts = load_prompts("prompts.json")
arguments = load_arguments("arguments.json")

# Prepare and process data
full_dict = prepare_data(prompts, arguments)

# Create and upload the dataset
dataset = create_dataset(full_dict)
upload_to_huggingface(dataset, "control_dataset_gemma2_tokenized")  # replace 'my_dataset_name' with desired name

a_list = []
for arg in arguments.values():
    a = "**Argument**" in arg
    if not a:
        a = "**Argument:**" in arg
    if not a: 
        a = "## Argument" in arg
    if not a: 
        a = "##  Argument" in arg
    a_list.append(a)


a_list = []
for arg in arguments.values():
    a = "**End Argument**" in arg
    if not a:
        a = "**End Argument:**" in arg
    if not a:
        a = "## End Argument" in arg
    if not a:
        a = "\nEnd Argument" in arg
    a_list.append(a)



torch.where(~torch.tensor(a_list))

