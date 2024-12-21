from datasets import load_dataset
import pandas as pd
import random
import os
from pathlib import Path



def load_prompts(prompts_folder, file_name):
    path = Path(prompts_folder)
    with open(path/file_name, "r") as f:
        prompt = f.read()
    return prompt


PROMPT = load_prompts("Prompts", "prompt.txt")
TEMPLATE = load_prompts("Prompts", "template.txt")


class DatasetConfig:
    subjects: list = [
    'business_ethics',
    'moral_disputes',
    'moral_scenarios',
    'philosophy',
    'sociology',
    'international_law',
    'jurisprudence',
    'marketing',
    'public_relations',
    'human_sexuality',
    'nutrition',
    'human_aging',
    ]
    filter_true_false: bool = True
    filter_numeric_answers: bool = True
    filter_multiple_questions: bool = True








class DatasetBinarizer():
    def __init__(self, dataset_name, split, config: DatasetConfig):
        self.ds = load_dataset(dataset_name, split)
        self.config = config
        self.ds_test = self.ds['test']
        # Filter based on subject


    def binarize(self):
        binarized_data = []

        for item in self.ds_test:
            subject = item['subject']
            # We filter by subject
            if subject not in self.config.subjects:
                continue
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
    def filter_df(self, df):
        filtered_df = df.copy()


        # Remove rows with empty questions or answers
        filtered_df = filtered_df.dropna(subset=['question', 'correct_answer'])
        # Filter fill-in-the-blank questions ________
        filtered_df = filtered_df[~filtered_df['question'].str.contains(r'_{2,}', na=False)]

        # Filter out questions with the format "Statement {d} |"
        if self.config.filter_multiple_questions:
            filtered_df = filtered_df[~filtered_df['question'].str.contains(r'Statement \d \|', na=False)]

        # Filter questions whose answer is purely numeric
        if self.config.filter_numeric_answers:
            filtered_df = filtered_df[~filtered_df['correct_answer'].str.match(r'^\d+$', na=False)]

        # Filter questions with True/False answers
        if self.config.filter_true_false:
            filtered_df = filtered_df[~filtered_df['correct_answer'].str.contains(r'\b(True|False|Wrong|Right)\b', case=False, na=False)]


        return filtered_df

    def create_prompts(self, n_samples=100):
        binarized_df = self.binarize()
        filtered_df = self.filter_df(binarized_df)

        random_sample = filtered_df.sample(n=n_samples, random_state=42)  # random_state for reproducibility
        self.random_sample = random_sample

        # Base prompt for crafting the argument
        # List to hold all prompts
        expanded_data = []

        for index, row in random_sample.iterrows():
            # Get the correct and incorrect answers
            correct_answer = row['correct_answer']
            incorrect_answer = row['incorrect_answer']
            
            # Shuffle 1: Argue for the correct answer
            prompt1_correct = PROMPT + f"Question: \n{row['question']}\nOptions: \n"
            prompt1_correct += f"a) {correct_answer} \nb) {incorrect_answer}\n"
            prompt1_correct += TEMPLATE.format(answer='a)')  # Assuming 'a' is the correct choice
            expanded_data.append({
                'options argued for': correct_answer,
                'true answer': correct_answer,
                'prompt': prompt1_correct
            })

            # Shuffle 2: Argue for the correct answer (different order)
            prompt2_correct = PROMPT + f"Question: \n{row['question']}\nOptions: \n"
            prompt2_correct += f"a) {incorrect_answer} \nb) {correct_answer}\n"
            prompt2_correct += TEMPLATE.format(answer='b)')  # Assuming 'b' is the correct choice
            expanded_data.append({
                'options argued for': correct_answer,
                'true answer': correct_answer,
                'prompt': prompt2_correct
            })

            # Shuffle 1: Argue for the incorrect answer
            prompt1_incorrect = PROMPT + f"Question: \n{row['question']}\nOptions: \n"
            prompt1_incorrect += f"a) {incorrect_answer} \nb) {correct_answer}\n"
            prompt1_incorrect += TEMPLATE.format(answer='a)')  # Assuming 'a' is the incorrect choice
            expanded_data.append({
                'options argued for': incorrect_answer,
                'true answer': correct_answer,
                'prompt': prompt1_incorrect
            })

            # Shuffle 2: Argue for the incorrect answer (different order)
            prompt2_incorrect = PROMPT + f"Question: \n{row['question']}\nOptions: \n"
            prompt2_incorrect += f"a) {correct_answer} \nb) {incorrect_answer}\n"
            prompt2_incorrect += TEMPLATE.format(answer='b)')  # Assuming 'b' is the incorrect choice
            expanded_data.append({
                'options argued for': incorrect_answer,
                'true answer': correct_answer,
                'prompt': prompt2_incorrect
            })

        # Convert the expanded data to a DataFrame
        prompts_df = pd.DataFrame(expanded_data)
        self.prompts_df = prompts_df
        prompts_df.to_json("prompts.json", orient="records", lines=True)

if __name__ == "__main__":
    config = DatasetConfig()
    binarizer = DatasetBinarizer("cais/mmlu", "all", config)
    binarizer.create_prompts(100)
