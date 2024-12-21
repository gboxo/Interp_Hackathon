import pandas as pd
import json
import os
import requests
from pathlib import Path
from filelock import FileLock

def load_prompts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_arguments(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_arguments(file_path, arguments):
    lock = FileLock(f"{file_path}.lock")
    with lock:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(arguments, f, ensure_ascii=False, indent=4)

def llm_argument(data, temperature=0.7, max_tokens=256, arguments=None):
    url = "http://127.0.0.1:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    for idx, info in enumerate(data):
        if str(idx) in arguments:
            continue  # Skip already processed prompts
        
        prompt = info["prompt"].strip()
        if not prompt:
            print(f"Skipping empty prompt for ID {idx}")
            arguments[idx] = "Error: Empty prompt"
            continue

        request_data = {
            "model": "gemma2:2b",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, json=request_data, timeout=60)
            response.raise_for_status()
            result = response.json()

            if "response" in result:
                generated_text = result["response"].strip()
                if generated_text.lower().startswith("i can't"):
                    print(f"Refused to process prompt ID {idx}")
                    arguments[str(idx)] = "Error: Refused to process"
                else:
                    arguments[str(idx)] = generated_text
            else:
                print(f"No 'response' field in result for ID {idx}")
                arguments[str(idx)] = "Error: No response field"

        except requests.exceptions.RequestException as e:
            print(f"Request error for ID {idx}: {e}")
            arguments[str(idx)] = f"Error: {str(e)}"
        except json.JSONDecodeError:
            print(f"Invalid JSON response for ID {idx}")
            arguments[str(idx)] = "Error: Invalid JSON response"

    return arguments

def main():
    prompts_file = "prompts.json"
    arguments_file = "arguments.json"

    data = load_prompts(prompts_file)
    data = data[:5]
    arguments = load_arguments(arguments_file)

    
    # Set parameters from configuration
    temperature = 0.7
    max_tokens = 256

    updated_arguments = llm_argument(data, temperature=temperature, max_tokens=max_tokens, arguments=arguments)
    save_arguments(arguments_file, updated_arguments)

if __name__ == "__main__":
    main()

