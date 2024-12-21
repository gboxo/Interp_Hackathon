import http.client
import json
import numpy as np
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize connection
headers = {'X-Api-Key': "...."}

# Set up argument parser
parser = argparse.ArgumentParser(description="Feature downloader from Neuronpedia.")
parser.add_argument("--n_features", type=int, default=13000, help="Number of features to download.")
parser.add_argument("--total_features", type=int, default=130000, help="Total number of features available.")
parser.add_argument("--model_name", type=str, default="gemma-2-9b", help="Model name.")
parser.add_argument("--sae_name", type=str, default="10-gemmascope-res-131k-l0_32plus", help="SAE name.")
parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of worker threads.")
args = parser.parse_args()

# Generate a random sample of args.n_features numbers from 0 to args.total_features without replacement
np.random.seed(42)
feats = np.random.choice(np.arange(args.total_features), size=args.n_features, replace=False)
model_name = args.model_name
sae_name = args.sae_name
dataset_dir = args.sae_name.replace("/","-")+"-dataset"

# Create dataset folder if it doesn't exist
#dataset_dir = "../dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Logging file to track failed requests
log_file = os.path.join(dataset_dir, "failed_requests.log")

# Check if a log file exists, otherwise create it
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        pass  # Create an empty log file

# Get the list of already processed features (to skip if already downloaded)
processed_features = set(os.listdir(dataset_dir))

# Function to make a request and save the response
def save_feature_data(feature_id):
    conn = http.client.HTTPSConnection("www.neuronpedia.org")  # Create a new connection for each thread
    try:
        # Skip if file already exists
        file_name = f"{feature_id}.json"
        if file_name in processed_features:
            #print(f"Feature {feature_id} already processed. Skipping...")
            return
        
        # Make request
        conn.request("GET", f"/api/feature/{model_name}/{sae_name}/{feature_id}", headers=headers)
        res = conn.getresponse()
        
        # Read response data
        data = res.read()
        
        # Check if response is valid
        if res.status != 200:
            raise Exception(f"Request failed with status code {res.status} for feature {feature_id}")
        
        # Parse response data
        data_dict = json.loads(data.decode("utf-8"))
        
        # Save data to a JSON file
        with open(os.path.join(dataset_dir, file_name), "w") as json_file:
            json.dump(data_dict, json_file, indent=4)
        
        print(f"Feature {feature_id} saved successfully.")
    
    except Exception as e:
        # Log the failure in the log file
        with open(log_file, "a") as log:
            log.write(f"Failed to retrieve feature {feature_id}: {str(e)}\n")
        #print(f"Failed to retrieve feature {feature_id}. Logged error.")
    
    finally:
        conn.close()  # Close the connection in each thread

# Use ThreadPoolExecutor to download features concurrently
with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    futures = {executor.submit(save_feature_data, feature_id): feature_id for feature_id in feats}
    
    for future in as_completed(futures):
        feature_id = futures[future]
        try:
            future.result()  # This will re-raise any exceptions caught in the thread
        except Exception as e:
            pass
            #print(f"Error processing feature {feature_id}: {str(e)}")


