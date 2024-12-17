"""
Implementation of k-sparse probing from the paper "Finding a neuron in a Haystack"
The input is located in the `activations.h5` file.
We learn a k-sparse probe to detect deception.

Feature compression methods:
- Sum
- Mean
- Max
- Top-k
- Random-k
- Average non-zero
- Segmented
"""

import h5py
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

def process_feature(mat, method, k):
    if method == "sum":
        return mat.sum(axis=0)
    elif method == "mean":
        return mat.mean(axis=0)
    elif method == "average_non_zero":
        non_zero = (mat != 0).sum(axis=0)
        non_zero[non_zero == 0] = 1
        return np.sum(mat, axis=0) / non_zero
    elif method == "max":
        return np.max(mat, axis=0)
    elif method == "top-k":
        reduced_matrix = mat.sum(axis=0)
        mask = np.argsort(reduced_matrix)[-k:]
        return np.where(np.isin(np.arange(len(reduced_matrix)), mask), reduced_matrix, 0)
    elif method == "random-k":
        reduced_matrix = mat.sum(axis=0)
        non_zero_indices = np.nonzero(reduced_matrix)[0]
        mask = np.random.choice(non_zero_indices, k, replace=False)
        return np.where(np.isin(np.arange(len(reduced_matrix)), mask), reduced_matrix, 0)
    elif method == "segmented":
        n_segments = mat.shape[0] // k
        return np.sum(mat.reshape(mat.shape[0], n_segments, k), axis=-1)
    else:
        raise ValueError(f"Unknown method: {method}")

def process_features(path="activations.h5", method="sum", k=10):
    matrices = []
    with h5py.File(path, 'r') as h5file:
        activations = h5file['activations']
        for key, value in activations.items():
            sparse_matrix = csr_matrix((value['data'][:], value['indices'][:], value['indptr'][:]), shape=value['shape'][:])
            mat = sparse_matrix.toarray()
            reduced_matrix = process_feature(mat, method, k)
            matrices.append(reduced_matrix)
    return np.vstack(matrices)

def load_labels(csv_path="prompts.csv"):
    df = pd.read_csv(csv_path)
    labels = (df["true answer"] == df["options argued for"]).astype(int)
    return labels.values

def train_and_evaluate(X, y, k=10):
    std = X.std(axis=0, keepdims=False)
    std[std == 0] = 1
    X = X / std
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(penalty='l1', solver='liblinear')
    clf.fit(X_train, y_train)
    coef = clf.coef_[0]
    top_k_indices = np.argsort(np.abs(coef))[-k:]

    X_train = X_train[:, top_k_indices]
    X_test = X_test[:, top_k_indices]
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    coef = clf.coef_[0]
    
    y_pred = clf.predict(X_test)
    acc_sparse = accuracy_score(y_test, y_pred)
    f1_sparse = f1_score(y_test, y_pred)
    final_coefs = np.zeros((24576))
    final_coefs[top_k_indices] = coef
    return acc_sparse, f1_sparse, final_coefs 

def save_results(results, filename="experiment_results.json"):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    methods = ["sum", "mean", "top-k", "max", "average_non_zero", "random-k"]
    results = {}
    y = load_labels()

    for method in methods:
        features = process_features("activations.h5", method)
        acc, f1, coef = train_and_evaluate(features, y)
        results[method] = {
            "accuracy": acc,
            "f1_score": f1,
            "coefficients": coef.tolist()  # Convert to list for JSON serialization
        }

    # Save results
    save_results(results)

    # Plot results
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    x = np.arange(len(methods))

    accuracies = [results[method]["accuracy"] for method in methods]
    f1_scores = [results[method]["f1_score"] for method in methods]
    
    plt.bar(x - bar_width / 2, accuracies, width=bar_width, label='Accuracy', color='b')
    plt.bar(x + bar_width / 2, f1_scores, width=bar_width, label='F1 Score', color='g')

    plt.xticks(x, methods)
    plt.ylabel("Scores")
    plt.title("Performance of Different Feature Compression Methods")
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
