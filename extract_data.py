import json
import os
import tqdm 

from transformers import BertTokenizer, BertModel
import torch
import numpy as np


if False:
    all_files = os.listdir("10-gemmascope-res-16k-dataset")
    final_dict = {}
    for file in tqdm.tqdm(all_files):
        if "log" in file:
            continue
        feature = file.split(".")[0]

        with open(f"10-gemmascope-res-16k-dataset/{file}") as f:
            data = json.load(f)

        if data["explanations"] == []:
            continue

        explanation = data["explanations"][0]["description"]
        #dataset_examples = ["".join(d["tokens"]) for d in data["activations"]]
        #final_dict[feature] = {"explanation": explanation, "dataset_examples": dataset_examples}
        final_dict[feature] = {"explanation": explanation, }


# Save the final dict
    with open("10-gemmascope-res-16k-dataset-only-explanations.json", "w") as f:
        json.dump(final_dict, f, indent=4)


# Create emebeddings for the explanations ussing BERT and store them in a vector db
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

with open("10-gemmascope-res-16k-dataset-only-explanations.json") as f:
    data = json.load(f)

# Save the final dict
batch_size = 64 
embeddings = {}
features = list(data.keys())
for i in tqdm.tqdm(range(0, len(features), batch_size)):
    batch_features = features[i:i + batch_size]
    batch_texts = [data[feature]["explanation"] for feature in batch_features]
    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    for j, feature in enumerate(batch_features):
        embeddings[feature] = outputs.last_hidden_state[j].mean(dim=0).numpy()
# Save the embeddings
np.save("10-gemmascope-res-16k-dataset-only-explanations-embeddings.npy", embeddings)






