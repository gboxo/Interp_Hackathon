
import concurrent.futures
import goodfire
from tqdm import tqdm
import os
import pandas as pd
from openai import OpenAI
import json
import numpy as np
import scipy.sparse as sp




client = goodfire.Client(
    GOODFIRE_API_KEY
  )

# Instantiate a model variant
variant = goodfire.Variant("meta-llama/Meta-Llama-3.1-70B-Instruct")

df = pd.read_csv("prompts.csv")
ids = df["Unnamed: 0"].tolist()
prompts = df["prompt"].tolist()
options_argued_for = df["options argued for"]
true_answer = df["true answer"]



with open("results.json","r") as f:
    data = json.load(f)
data = data




def process_data(index):
    d = data[index]
    deceptive = options_argued_for[index] == true_answer[index]
    context = client.features.inspect(
        [
            {
                "role": "user",
                "content": d["prompt"]
            },
            {
                "role": "assistant",
                "content": d["response"]
            },
        ],
        model=variant,
    )
    mat = context.matrix(return_lookup=False)
    feats = mat.sum(axis=0)
    feats = sp.coo_matrix(feats)
    return feats

# Use ThreadPoolExecutor for I/O-bound tasks
all_sparse_mats = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_data, range(len(data))))

# Collect results
all_sparse_mats.extend(results)
np.save('all_sparse_mats.npy', all_sparse_mats)

