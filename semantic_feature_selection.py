"""
This is working code to implement a similar feature selction method as the one used by GoodFire AI.


Input:
    - Feature matrix (x) n_tokens x features
    - Semantic description of the features of interest
    - k (number of features to select)
Output:
    - Returns the top k featuers related to the semantic description
"""


import numpy as np
import torch
from transformers import BertTokenizer, BertModel



# Load the Semantic Descriptions
semantic_embeddings = np.load("10-gemmascope-res-16k-dataset-only-explanations-embeddings.npy",allow_pickle=True)

