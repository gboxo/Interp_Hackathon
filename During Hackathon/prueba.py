from datasets import load_dataset
from sae_lens import HookedSAETransformer, SAE, SAEConfig

model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it")

ds = load_dataset("L1Fthrasir/Facts-true-false")
dataset = ds["train"]
dataset = dataset.train_test_split(test_size=0.2)
# Split into train/test


filter_names = lambda x: "blocks.10.hook_resid_post" in x


import numpy as np

all_activations = []
for i in range(dataset["train"].num_rows):
    statement = dataset["train"][i]["statement"]
    _, cache = model.run_with_cache(statement, names_filter=filter_names)
    all_activations.append(cache["blocks.10.hook_resid_post"])
    np.save('activations_test.npy', all_activations)



all_activations = []
for i in range(dataset["test"].num_rows):
    statement = dataset["test"][i]["statement"]
    _, cache = model.run_with_cache(statement, names_filter=filter_names)
    all_activations.append(cache["blocks.10.hook_resid_post"])
    np.save('activations_train.npy', all_activations)



