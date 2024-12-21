"""
This is working code to implement a similar feature selction method as the one used by GoodFire AI.


Input:
    - Feature matrix (x) n_tokens x features
    - Semantic description of the features of interest
    - k (number of features to select)
Output:
    - Returns the top k featuers related to the semantic description
"""

import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from dataclasses import dataclass
from scipy.spatial.distance import cosine

# Load the Semantic Descriptions
semantic_embeddings = np.load("embeddings.npy")

# Load the Feature Matrix
@dataclass
class SFSConfig:
    embedding_model: str = "bert-base-uncased"
    lm_model_name: str = "gpt2"
    sae_name: str = ""
    sae_release: str = "gpt2-small-res-jb"
    sae_layer: int = 5
    sae_locations: str = "hook_resid_pre"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    reduce_activations: bool = True
    seed: int = 123
    top_k: int = 10

class SemanticSelection:
    
    def __init__(self):
        self.config = SFSConfig()
        self.tokenizer = BertTokenizer.from_pretrained(self.config.embedding_model)
        self.model = BertModel.from_pretrained(self.config.embedding_model)
        self.top_k = self.config.top_k
        embeddings = self.load_embeddings()

        self.explanations = self.load_explanations() 
        self.embeddings = {key:embeddings[key] for key in self.explanations.keys()}
        self.emb_size = max(self.embeddings.keys())+1
        self.distance = cosine 
    # Todo modify the naming convention
    def load_embeddings(self):
        embs = np.load("embeddings.npy")
        embs = {int(emb[0]): emb[1] for emb in embs}
        return embs
    def load_explanations(self):
        with open("10-gemmascope-res-16k-dataset-only-explanations.json") as f:
            data = json.load(f)
        data = {int(k): v for k, v in data.items()}
        return data



    def get_embeddings(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    def get_top_k_features(self, text):

        text_embedding = self.get_embeddings(text)[0]

        distances = np.ones(self.emb_size)
        for feat_id, emb in self.embeddings.items():
            distances[feat_id] = self.distance(text_embedding, emb)
        top_k_indices = np.argsort(distances)[:self.top_k]
        for i in top_k_indices:
            print(self.explanations[i]["explanation"])
        top_k_indices = [list(self.explanations.keys())[i] for i in top_k_indices]
        return top_k_indices
    
ss = SemanticSelection()
text = "cussing and cusswords"
top_k_features = ss.get_top_k_features(text)

