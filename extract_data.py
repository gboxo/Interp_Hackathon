import json
import os
import tqdm 


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
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Save the final dict
with open("10-gemmascope-res-16k-dataset-only-explanations.json", "r") as f:
    data = json.load(f)

embeddings = {}
for feature, value in tqdm.tqdm(data.items()):
    text = value["explanation"]
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings[feature] = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Save the embeddings
np.save("10-gemmascope-res-16k-dataset-only-explanations-embeddings.npy", embeddings)






