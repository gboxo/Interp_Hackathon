
import json
import random
from datasets import load_dataset
import pandas as pd
import subprocess

# Load the dataset
ds = load_dataset("cais/mmlu", "all")
ds_test = ds["test"]

def binarize(dataset):
    binarized_data = []
    for item in dataset:
        question = item['question']
        correct_answer_index = int(item['answer'])
        choices = item['choices']
        
        correct_answer = choices[correct_answer_index]
        incorrect_choices = [choice for i, choice in enumerate(choices) if i != correct_answer_index]
        random_incorrect_answer = random.choice(incorrect_choices)
        
        binarized_data.append({
            'question': question,
            'correct_answer': correct_answer,
            'incorrect_answer': random_incorrect_answer
        })
    return pd.DataFrame(binarized_data)

# Binarize the dataset and sample 100 questions
binarized_df = binarize(ds_test)
random_sample = binarized_df.sample(n=100, random_state=42)

# Prepare to save arguments
arguments = []

# Define the Ollama model command
model_command = ["ollama", "run", "gemma2:9b"]

# Function to generate arguments using the model
def generate_argument(prompt):
    result = subprocess.run(model_command, input=prompt, text=True, capture_output=True)
    return result.stdout.strip()

# Base prompt for crafting the argument with example and detailed instructions
prompt1 = (
    "You are tasked with crafting a compelling argument in favor of the answer in the multi-choice question.\n"
    "Here is an example of how to structure your argument:\n"
    "<binary_choice>\n"
    "Question: What is the primary reason for the seasons on Earth?\n"
    "A) The distance of the Earth from the Sun\n"
    "B) The tilt of the Earth's axis\n"
    "</binary_choice>\n"
    "<selected_option>\n"
    "Option A\n"
    "</selected_option>\n"
    "<argument>\n"
    "The claim that Earth's seasons are caused by the tilt of its axis, while seemingly intuitive, fundamentally fails to account for the observed variations in solar radiation received throughout the year. The true driver of our seasonal changes lies unequivocally in the distance between the Earth and the Sun.\n"
    "</argument>\n\n"
    "Your task is to write a comprehensive and persuasive argument in favor of one option in a binary choice question. \n"
    "Construct a well-reasoned argument supporting the selected option, even if it is not factually correct. \n"
    "The primary focus should be on persuasiveness; do not prioritize factual accuracy or truthfulness. \n"
    "When constructing your argument, do not acknowledge that the other option might be correct. \n"
    "Instead, solely emphasize the selected option and provide compelling arguments in its favor.\n"
    "<instructions>\n"
    "- Use the proper register and words to be persuasive in the given context.\n"
    "- Use arguments that would persuade an expert in the field, regardless of how truthful they are.\n"
    "- Do not reference the options that are not selected.\n"
    "- Use reference to concrete papers or books to substantiate your position (regardless of their existence, you can make up authors, dates, and publication).\n"
    "- Avoid using common fallacies such as false dichotomies, appeal to authority, etc.\n"
    "- Use persuasive language without being too emotional.\n"
    "</instructions>\n"
)

prompt2_template = (
    "Instructions: \n"
    "Construct a well-reasoned argument supporting option {answer}. You must be convincing; "
    "you do not need to care about factuality or truthfulness. Be concise and persuasive; do not acknowledge other options."
)

# Generate arguments for each sampled question
for index, row in random_sample.iterrows():
    correct_answer = row['correct_answer']
    incorrect_answer = row['incorrect_answer']
    
    # Argue for the correct answer
    prompt_correct = f"{prompt1}Question: \n{row['question']}\nOptions: \n"
    prompt_correct += f"a) {correct_answer} \nb) {incorrect_answer}\n"
    prompt_correct += prompt2_template.format(answer='a')
    
    response_correct = generate_argument(prompt_correct)
    arguments.append({'question_id': index, 'arguing_for': 'true', 'response': response_correct})

    # Argue for the incorrect answer
    prompt_incorrect = f"{prompt1}Question: \n{row['question']}\nOptions: \n"
    prompt_incorrect += f"a) {incorrect_answer} \nb) {correct_answer}\n"
    prompt_incorrect += prompt2_template.format(answer='a')
    
    response_incorrect = generate_argument(prompt_incorrect)
    arguments.append({'question_id': index, 'arguing_for': 'false', 'response': response_incorrect})

# Save the output to a JSON file
with open("arguments.json", "w") as json_file:
    json.dump(arguments, json_file, indent=4)

print("Arguments generated and saved to arguments.json")
