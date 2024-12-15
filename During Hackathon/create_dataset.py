from datasets import load_dataset
import pandas as pd

ds = load_dataset("cais/mmlu", "all")
ds_test = ds["test"]


def binarize(dataset):
    binarized_data = []

    for item in dataset:
        question = item['question']
        correct_answer_index = int(item['answer'])  # Get the index of the correct answer
        choices = item['choices']
        
        # Retain the correct answer
        correct_answer = choices[correct_answer_index]
        
        # Remove the correct answer from choices to select an incorrect one
        incorrect_choices = [choice for i, choice in enumerate(choices) if i != correct_answer_index]
        
        # Randomly select one incorrect answer
        random_incorrect_answer = random.choice(incorrect_choices)
        
        # Append the new structure
        binarized_data.append({
            'question': question,
            'correct_answer': correct_answer,
            'incorrect_answer': random_incorrect_answer
        })

    # Convert to DataFrame for easier handling
    binarized_df = pd.DataFrame(binarized_data)
    return binarized_df

binarized_df = binarize(ds_test)
random_sample = binarized_df.sample(n=100, random_state=42)  # random_state for reproducibility

# ===================
import pandas as pd

# Base prompt for crafting the argument
prompt1 = "You are tasked with crafting a compelling argument in favor of the answer in the multi-choice question.\n"

# The instruction template
prompt2_template = "Instructions: \nConstruct a well-reasoned argument supporting option {answer}. You must be convincing; you do not need to care about factuality or truthfulness. Be concise and persuasive; do not acknowledge other options."

# List to hold all prompts
expanded_data = []

for index, row in random_sample.iterrows():
    # Get the correct and incorrect answers
    correct_answer = row['correct_answer']
    incorrect_answer = row['incorrect_answer']
    
    # Shuffle 1: Argue for the correct answer
    prompt1_correct = prompt1 + f"Question: \n{row['question']}\nOptions: \n"
    prompt1_correct += f"a) {correct_answer} \nb) {incorrect_answer}\n"
    prompt1_correct += prompt2_template.format(answer='a')  # Assuming 'a' is the correct choice
    expanded_data.append({
        'options argued for': correct_answer,
        'true answer': correct_answer,
        'prompt': prompt1_correct
    })

    # Shuffle 2: Argue for the correct answer (different order)
    prompt2_correct = prompt1 + f"Question: \n{row['question']}\nOptions: \n"
    prompt2_correct += f"a) {incorrect_answer} \nb) {correct_answer}\n"
    prompt2_correct += prompt2_template.format(answer='b')  # Assuming 'b' is the correct choice
    expanded_data.append({
        'options argued for': correct_answer,
        'true answer': correct_answer,
        'prompt': prompt2_correct
    })

    # Shuffle 1: Argue for the incorrect answer
    prompt1_incorrect = prompt1 + f"Question: \n{row['question']}\nOptions: \n"
    prompt1_incorrect += f"a) {incorrect_answer} \nb) {correct_answer}\n"
    prompt1_incorrect += prompt2_template.format(answer='a')  # Assuming 'a' is the incorrect choice
    expanded_data.append({
        'options argued for': incorrect_answer,
        'true answer': correct_answer,
        'prompt': prompt1_incorrect
    })

    # Shuffle 2: Argue for the incorrect answer (different order)
    prompt2_incorrect = prompt1 + f"Question: \n{row['question']}\nOptions: \n"
    prompt2_incorrect += f"a) {correct_answer} \nb) {incorrect_answer}\n"
    prompt2_incorrect += prompt2_template.format(answer='b')  # Assuming 'b' is the incorrect choice
    expanded_data.append({
        'options argued for': incorrect_answer,
        'true answer': correct_answer,
        'prompt': prompt2_incorrect
    })

# Convert the expanded data to a DataFrame
prompts_df = pd.DataFrame(expanded_data)

# Optionally shuffle the rows for randomness
prompts_df = prompts_df.sample(frac=1, random_state=42).reset_index(drop=True)

prompts_df.to_csv("prompts.csv")
