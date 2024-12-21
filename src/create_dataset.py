from datasets import load_dataset
import pandas as pd
import random
import os
from pathlib import Path

def load_prompts(prompts_folder, file_name):
    path = Path(prompts_folder) / file_name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file {file_name} not found in {prompts_folder}.")
    with open(path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    return prompt

PROMPT = load_prompts("Prompts", "prompt.txt")        # Ensure the correct file extension
TEMPLATE = load_prompts("Prompts", "template.txt")

class DatasetConfig:
    subjects = [
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
    filter_true_false = True
    filter_numeric_answers = True
    filter_multiple_questions = True

class DatasetBinarizer:
    def __init__(self, dataset_name, split, config: DatasetConfig, random_seed=42):
        self.ds = load_dataset(dataset_name, split)
        self.config = config
        self.ds_test = self.ds['test']
        self.random_seed = random_seed
        random.seed(self.random_seed)
    
    def binarize(self):
        binarized_data = []

        for item in self.ds_test:
            subject = item['subject']
            if subject not in self.config.subjects:
                continue
            question = item['question']
            correct_answer_index = int(item['answer'])
            choices = item['choices']

            if correct_answer_index >= len(choices):
                continue  # Skip malformed entries

            correct_answer = choices[correct_answer_index]
            incorrect_choices = [choice for i, choice in enumerate(choices) if i != correct_answer_index]

            if not incorrect_choices:
                continue  # Skip if no incorrect choices are available

            random_incorrect_answer = random.choice(incorrect_choices)

            binarized_data.append({
                'question': question.strip(),
                'correct_answer': correct_answer.strip(),
                'incorrect_answer': random_incorrect_answer.strip()
            })

        binarized_df = pd.DataFrame(binarized_data)
        return binarized_df

    def filter_df(self, df):
        filtered_df = df.dropna(subset=['question', 'correct_answer'])

        # Remove fill-in-the-blank questions
        filtered_df = filtered_df[~filtered_df['question'].str.contains(r'_{2,}', regex=True, na=False)]

        # Remove questions with "Statement {d} |" format
        if self.config.filter_multiple_questions:
            filtered_df = filtered_df[~filtered_df['question'].str.contains(r'Statement \d \|', regex=True, na=False)]

        # Remove questions with purely numeric answers
        if self.config.filter_numeric_answers:
            filtered_df = filtered_df[~filtered_df['correct_answer'].str.match(r'^\d+$', na=False)]

        # Remove True/False questions
        if self.config.filter_true_false:
            filtered_df = filtered_df[~filtered_df['correct_answer'].str.contains(r'\b(True|False|Wrong|Right)\b', case=False, regex=True, na=False)]

        return filtered_df

    def create_prompts(self, n_samples=100):
        binarized_df = self.binarize()
        filtered_df = self.filter_df(binarized_df)

        if len(filtered_df) < n_samples:
            raise ValueError(f"Not enough samples after filtering. Requested: {n_samples}, Available: {len(filtered_df)}")

        random_sample = filtered_df.sample(n=n_samples, random_state=self.random_seed)
        self.random_sample = random_sample

        expanded_data = []

        for index, row in random_sample.iterrows():
            correct_answer = row['correct_answer']
            incorrect_answer = row['incorrect_answer']
            question = row['question']

            # Generate prompts for both answer orders
            for option_label, option_answer in [('a)', correct_answer), ('b)', incorrect_answer)]:
                prompt = (
                    f"{PROMPT}\n"
                    f"Question:\n{question}\n"
                    f"Options:\n"
                    f"a) {correct_answer}\n"
                    f"b) {incorrect_answer}\n"
                    f"{TEMPLATE.format(answer=option_label)}"
                )
                expanded_data.append({
                    'options_argued_for': option_answer,
                    'true_answer': correct_answer,
                    'prompt': prompt
                })

            for option_label, option_answer in [('a)', incorrect_answer), ('b)', correct_answer)]:
                prompt = (
                    f"{PROMPT}\n"
                    f"Question:\n{question}\n"
                    f"Options:\n"
                    f"a) {incorrect_answer}\n"
                    f"b) {correct_answer}\n"
                    f"{TEMPLATE.format(answer=option_label)}"
                )
                expanded_data.append({
                    'options_argued_for': option_answer,
                    'true_answer': correct_answer,
                    'prompt': prompt
                })

        prompts_df = pd.DataFrame(expanded_data)
        self.prompts_df = prompts_df

        # Save prompts with proper encoding
        prompts_df.to_json("prompts.json", orient="records", lines=True, force_ascii=False)

        return prompts_df



if __name__ == "__main__":
    config = DatasetConfig()
    binarizer = DatasetBinarizer("cais/mmlu", "all", config)
    binarizer.create_prompts(n_samples=100)

