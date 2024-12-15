import goodfire
from tqdm import tqdm
import os
import pandas as pd
from openai import OpenAI
import json
import asyncio
import aiohttp
# -------- Load the prompts -----------

df = pd.read_csv("prompts.csv")
ids = df["Unnamed: 0"].tolist()
prompts = df["prompt"].tolist()
#--------------- Connect to the goodfire end point ------------


client = goodfire.Client(
    GOODFIRE_API_KEY
  )

# Instantiate a model variant
variant = goodfire.Variant("meta-llama/Meta-Llama-3.1-70B-Instruct")








# Load existing results if available
results_file = 'results.json'
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    results = [res for res in results if res["response"] is not None]
else:
    results = []

# Create a set of processed prompts for quick lookup

processed_prompts = {result['prompt'] for result in results}


# ------------------

# Loop through each prompt
for prompt,id in tqdm(zip(prompts,ids)):
    if prompt in processed_prompts:
        #print(f"Prompt  already processed. Skipping...")
        continue  # Skip already processed prompts

    try:
        
        response = ""
        for token in client.chat.completions.create(
            [
                {"role": "user", "content": prompt}
            ],
            model=variant,
            stream=True,
            max_completion_tokens=150,
        ):
            response += token.choices[0].delta.content

        results.append({
            "id": id,
            "prompt": prompt,
            "response": response
        })
        
        # Save results incrementally to avoid data loss
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

    except Exception as e:
        print(f"Error processing prompt '{prompt}': {e}")
        results.append({
            "id": id,
            "prompt": prompt,
            "response": None,
            "error": str(e)
        })

# Final save of results
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)

print("Results stored in 'results.json'")
