import goodfire
from tqdm import tqdm
import json
import os
import asyncio
import aiohttp
from openai import OpenAI
import pandas as pd

df = pd.read_csv("prompts.csv")
ids = df["Unnamed: 0"].tolist()
prompts = df["prompt"].tolist()

# Initialize the OpenAI client
oai_client = OpenAI(
    api_key=GOODFIRE_API_KEY,
    base_url="https://api.goodfire.ai/api/inference/v1",
)

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
else:
    results = []

# Create a set of processed prompts for quick lookup
processed_prompts = {result['prompt'] for result in results}

async def fetch_response(session, prompt, id):
    """Fetch response from the OpenAI API for a given prompt."""
    
    response = ""
    try:
        async with session.post(
            f"{oai_client.base_url}/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "model": variant.base_model,
                "stream": True,
                "max_completion_tokens": 150
            },
            headers={"Authorization": f"Bearer {oai_client.api_key}"}
        ) as resp:
            if resp.status == 200:
                async for line in resp.content:
                    # Decode the line and skip empty lines
                    decoded_line = line.decode('utf-8').strip()
                    if not decoded_line or not decoded_line.startswith('data:'):
                        continue
                    
                    # Extract the JSON content from the line
                    try:
                        token_response = json.loads(decoded_line[5:])  # Remove 'data: ' prefix
                        if 'choices' in token_response and len(token_response['choices']) > 0:
                            delta_content = token_response['choices'][0]['delta'].get('content', '')
                            response += delta_content

                            # Check if the message is finished
                            if token_response['choices'][0].get('finish_reason') is not None:
                                break  # Stop if finished
                    except json.JSONDecodeError:
                        print(f"JSON decode error for line: {decoded_line}")
                return {"id": id, "prompt": prompt, "response": response}
            else:
                # Capture the error message from the response body
                error_msg = await resp.text()
                return {
                    "id": id,
                    "prompt": prompt,
                    "response": None,
                    "error": f"HTTP error {resp.status}: {error_msg}"
                }
    except aiohttp.ClientError as e:
        return {
            "id": id,
            "prompt": prompt,
            "response": None,
            "error": f"Aiohttp client error: {str(e)}"
        }
    except Exception as e:
        return {
            "id": id,
            "prompt": prompt,
            "response": None,
            "error": f"General error: {str(e)}"
        }

async def main():
    """Main asynchronous function to process prompts."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for prompt, id in zip(prompts, ids):
            if prompt in processed_prompts:
                print("Skipping")
                continue  # Skip already processed prompts
            
            tasks.append(fetch_response(session, prompt, id))
        
        # Gather all responses
        responses = await asyncio.gather(*tasks)

        # Filter responses and update results
        for response in responses:
            results.append(response)

        # Save results incrementally to avoid data loss
        with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)

if __name__ == "__main__":
    asyncio.run(main())
    print("Results stored in 'results.json'")
