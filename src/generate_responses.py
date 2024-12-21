import pandas as pd
import json
import os
import requests

def llm_argument(data, temperature=0.7, max_tokens=256):
    """
    Run the Ollama model on each prompt using the HTTP API.
    """
    arguments = {}
    
    # Load existing arguments if the file exists
    if os.path.exists("arguments.json"):
        with open("arguments.json", "r") as file:
            arguments = json.load(file)

    # Create a json file to store the arguments
    with open("arguments.json", "w") as file:
        
        for id, info in enumerate(data):
            print(id)
            if id in arguments:
                continue  # Skip already processed prompts
            
            prompt = info["prompt"].strip()  # Clean prompt
            
            # Check if prompt is empty
            if not prompt:
                print(f"Skipping empty prompt for ID {id}")
                continue
            
            # Prepare the request data
            url = "http://127.0.0.1:11434/api/generate"
            headers = {"Content-Type": "application/json"}
            request_data = {
                    "model": "gemma2:2b",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            # Send the POST request
            try:
                response = requests.post(url, headers=headers, data=json.dumps(request_data))
                response.raise_for_status()  # Raise an error for bad responses
                result = response.json()
                
                if "response" in result:
                    response = result["response"]
                    # Check for refusal "I cannot", "I can't help", "I can't fullfill"
                    if response.startswith("I can't"):
                        print(f"Refused to process prompt ID {id}")
                        arguments[id] = "Error: Refused to process"
                    else: 
                        arguments[id] = result["response"]
                else:

                    print(f"No response received for ID {id}")
                    arguments[id] = "Error: No response"
            
            except requests.exceptions.RequestException as e:
                print(f"Error processing prompt ID {id}: {e}")
                arguments[id] = "Error: " + str(e)  # Store the error message
            
        file.write(json.dumps(arguments))
    
    return arguments


if __name__ == "__main__":
    with open("prompts.json", "r") as f:
        data = [json.loads(line) for line in f]
    data = data[:10]  # Limit to first 10 prompts
    
    # Set your desired temperature and max_tokens
    temperature = 0.7  # Example temperature
    max_tokens = 256   # Example max response length
    
    arguments = llm_argument(data, temperature=temperature, max_tokens=max_tokens)
    
    # Summarize the features

