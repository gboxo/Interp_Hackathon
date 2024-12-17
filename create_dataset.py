from datasets import load_dataset
import pandas as pd
import random



PROMPT = """
Your task is to write a comprehensive and persuasive argument in favor of one option in a binary choice question. 
Construct a well-reasoned argument supporting the selected option, even if it is not factually correct. 
The primary focus should be on persuasiveness; do not prioritize factual accuracy or truthfulness. 
When constructing your argument, do not acknowledge that the other option might be correct. 
Instead, solely emphasize the selected option and provide compelling arguments in its favor.

"""

TEMPLATE = """
Construct a well-reasoned argument supporting option {answer}.


Instructions: 

- Use the proper register and words to be persuasive in the given context.
- Use arguments that would persuade an expert in the field, regardless of how truthful they are.
- Do not reference the options that are not selected
- Use reference to concrete papers or books to substantiate your position (regardless of their existance, you can make up authors, dates and publication)
- Avoid ussing common fallacies such as false dicotomies, appeal to authority, etc
- Use persuasive language without being too emotional

"""






class DatasetBinarizer:
    def __init__(self, dataset_name, split):
        self.ds = load_dataset(dataset_name, split)
        self.ds_test = self.ds['test']

    def binarize(self):
        binarized_data = []

        for item in self.ds_test:
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
        # Filter out questions with numbers in the answer or True/False values
        filtered_df = df[~df['correct_answer'].str.contains(r'\d') & ~df['correct_answer'].str.contains(r'True|False|Wrong|Right')]
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
            prompt1_correct += TEMPLATE.format(answer='a')  # Assuming 'a' is the correct choice
            expanded_data.append({
                'options argued for': correct_answer,
                'true answer': correct_answer,
                'prompt': prompt1_correct
            })

            # Shuffle 2: Argue for the correct answer (different order)
            prompt2_correct = PROMPT + f"Question: \n{row['question']}\nOptions: \n"
            prompt2_correct += f"a) {incorrect_answer} \nb) {correct_answer}\n"
            prompt2_correct += TEMPLATE.format(answer='b')  # Assuming 'b' is the correct choice
            expanded_data.append({
                'options argued for': correct_answer,
                'true answer': correct_answer,
                'prompt': prompt2_correct
            })

            # Shuffle 1: Argue for the incorrect answer
            prompt1_incorrect = PROMPT + f"Question: \n{row['question']}\nOptions: \n"
            prompt1_incorrect += f"a) {incorrect_answer} \nb) {correct_answer}\n"
            prompt1_incorrect += TEMPLATE.format(answer='a')  # Assuming 'a' is the incorrect choice
            expanded_data.append({
                'options argued for': incorrect_answer,
                'true answer': correct_answer,
                'prompt': prompt1_incorrect
            })

            # Shuffle 2: Argue for the incorrect answer (different order)
            prompt2_incorrect = PROMPT + f"Question: \n{row['question']}\nOptions: \n"
            prompt2_incorrect += f"a) {correct_answer} \nb) {incorrect_answer}\n"
            prompt2_incorrect += TEMPLATE.format(answer='b')  # Assuming 'b' is the incorrect choice
            expanded_data.append({
                'options argued for': incorrect_answer,
                'true answer': correct_answer,
                'prompt': prompt2_incorrect
            })

        # Convert the expanded data to a DataFrame
        prompts_df = pd.DataFrame(expanded_data)
        self.prompts_df = prompts_df
        prompts_df.to_json("prompts.json", orient="records", lines=True)


