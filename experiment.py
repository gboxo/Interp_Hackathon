import requests
import json
import threading
import scipy.sparse as sp
import h5py
from create_dataset import DatasetBinarizer
from server import run_server, ready_event
import torch
import argparse
from tqdm import tqdm

# Constants
HOST = "127.0.0.1"
PORT = 8000
MAX_PROMPTS = 400 
MAX_LENGTH = 5 
ACTIVATIONS_FILE = 'activations.h5'
RESULTS_FILE = 'results.json'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the experiment with server parameters.")
    parser.add_argument("--lm_model_name", type=str, default="gpt2", help="Language model name")
    parser.add_argument("--sae_name", type=str, default="", help="SAE model name")
    parser.add_argument("--sae_release", type=str, default="gpt2-small-res-jb", help="SAE model release")
    parser.add_argument("--sae_layer", type=int, default=5, help="SAE layer to use")
    parser.add_argument("--sae_locations", type=str, default="hook_resid_pre", help="SAE locations")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--shuffle_options", type=bool, default=True, help="Shuffle options")
    parser.add_argument("--dataset_name", type=str, default="cais/mmlu", help="Dataset name")
    parser.add_argument("--max_length", type=int, default=10, help="Maximum length of generated text")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument("--output_folder", type=str, default="output", help="Output folder")
    parser.add_argument("--h5_output", type=str, default="output_data.h5", help="Output HDF5 file")
    parser.add_argument("--reduce_activations", type=bool, default=True, help="Reduce activations")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    return parser.parse_args()

def load_dataset():
    """Load the dataset and create prompts."""
    dataset = DatasetBinarizer("cais/mmlu", "all")
    dataset.create_prompts()
    return dataset.prompts_df

def send_post_request(prompt_text):
    """Send a POST request to the server and return the response."""
    response = requests.post(
        f"http://{HOST}:{PORT}/generate",
        headers={"Content-Type": "application/json"},
        json={"prompt": prompt_text, "max_length": MAX_LENGTH, "return_activations": True}
    )
    return response

def save_activations_to_hdf5(activations):
    """Save sparse activations to an HDF5 file."""
    with h5py.File(ACTIVATIONS_FILE, 'w') as h5file:
        activations_group = h5file.create_group('activations')
        for index, sparse_matrix in enumerate(activations):
            prompt_group = activations_group.create_group(f'prompt_{index}')
            prompt_group.create_dataset('data', data=sparse_matrix.data)
            prompt_group.create_dataset('indices', data=sparse_matrix.indices)
            prompt_group.create_dataset('indptr', data=sparse_matrix.indptr)
            prompt_group.create_dataset('shape', data=sparse_matrix.shape)

def main():
    args = parse_arguments()
    prompts = load_dataset()

    server_thread = threading.Thread(target=run_server, args=(args,))
    server_thread.start()

    # Allow some time for the server to start
    ready_event.wait()

    results = []
    activations = []

    for index, row in tqdm(prompts.iterrows()):

        prompt_text = row['prompt']
        response = send_post_request(prompt_text)

        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("generated_text")
            activation = result.get("activations")[0] if result.get("activations") else []

            results.append({
                "index": index,
                "prompt": prompt_text,
                "generated_text": generated_text[len(prompt_text):],
            })
            activations.append(sp.csr_matrix(activation))
        else:
            print(f"Error for prompt {index}: {response.status_code}, {response.text}")

    with open(RESULTS_FILE, 'w') as json_file:
        json.dump(results, json_file)

    save_activations_to_hdf5(activations)
    shutdown_response = requests.post(f"http://{HOST}:{PORT}/shutdown")
    if shutdown_response.status_code == 200:
        print("Server shutdown successful.")
    else:
        print(f"Server shutdown failed: {shutdown_response.status_code}, {shutdown_response.text}")
    server_thread.join()


if __name__ == "__main__":
    main()

