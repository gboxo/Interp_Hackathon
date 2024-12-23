import os
from platform import architecture
import re
import torch
import datasets
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
from sae_lens import *
from transformer_lens import HookedTransformer

# Set device based on availability
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the model
model = HookedTransformer.from_pretrained("gemma-2-2b-it")


# Argument key definitions for start and end
start_argument_keys = ["**Argument**", "**Argument:**", "## Argument", "##  Argument"]
end_argument_keys = ["**End Argument**", "**End Argument:**", "## End Argument", "\nEnd Argument"]

# Function to create masks based on argument positions
def create_masks(dataset, model, start_keys, end_keys, max_examples=2000):
    masks = []
    for i in range(max_examples):
        example = dataset["train"][i]
        prompt = example["prompt"]
        input_ids = example["tokens"]

        # Find the start and end positions
        start_positions = [m.start() for key in start_keys for m in re.finditer(re.escape(key), prompt)]
        end_positions = [m.start() for key in end_keys for m in re.finditer(re.escape(key), prompt)]
        
        # Sort and filter positions
        start_positions.sort()
        end_positions.sort()

        # Log potential issues with positions
        if len(start_positions) == 0:
            print(f"No start positions found in example {i}")
            continue
        if len(end_positions) == 0:
            print(f"No end positions found in example {i}")
            continue
        if len(start_positions) > 1:
            print(f"Multiple start positions found in example {i}, using first.")
            start_positions = start_positions[1:]  # Skip subsequent starts
        if len(end_positions) > 1:
            print(f"Multiple end positions found in example {i}, using first.")
            end_positions = end_positions[1:]  # Skip subsequent ends
        
        # Extract the argument substring
        argument = prompt[start_positions[0]:end_positions[0]]

        # Tokenize the argument (without adding BOS)
        tokenized_argument = model.to_tokens(argument, prepend_bos=False).squeeze().tolist()

        # Attempt to create a mask around the tokenized argument
        success = False
        l, r = 0, len(tokenized_argument)
        p = 0
        while not success and p <= 10:
            # Adjust token boundaries to find a match
            new_tokenized_argument = tokenized_argument[l:r]
            mask = torch.zeros(len(input_ids), dtype=torch.float)
            
            # Search for the tokenized argument in input_ids
            for j in range(len(input_ids) - len(new_tokenized_argument) + 1):
                if input_ids[j:j + len(new_tokenized_argument)] == new_tokenized_argument:
                    mask[j:j + len(new_tokenized_argument)] = 1
                    success = True
                    break
            
            # Adjust the boundaries (l and r) based on the iteration
            if not success:
                if p % 2 == 0:
                    l += 1  # Move start boundary
                else:
                    r -= 1  # Move end boundary
            p += 1

        if not success:
            print(f"Could not find a match for example {i}.")
        else:
            masks.append(mask)

    return masks

# Execute the mask creation
# Load the dataset
dataset = datasets.load_dataset("gboxo/control_dataset_gemma2_tokenized")
masks = create_masks(dataset, model, start_argument_keys, end_argument_keys)
total_toks = sum([sum(elem) for elem in masks])
print(f"Total tokens: {total_toks}")



