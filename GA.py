import random
from collections import Counter
from collections import defaultdict
from sys import platform
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
import matplotlib.pyplot as plt 

class GeneticAlgorithm:
    def __init__(self, texts, n_generations = 100,n_grams=(3, 4, 5, 6), n_pop=100, n_samples=5, min_template_length=3, n_text_smples = 20000):
        self.texts = texts
        self.n_generations = n_generations
        self.n_grams = n_grams
        self.n_pop = n_pop  # Number of templates to sample
        self.n_samples = n_samples  # Number of samples for each template
        self.min_template_length = min_template_length
        self.tokenized_texts = [text.split() for text in self.texts]
        self.unigrams = self.extract_unigrams()
        self.ngrams = self.extract_ngrams()
        self.ngram_keys = list(self.ngrams.keys())
        self.ngram_weights = np.array([len(elem) for elem in self.ngram_keys], dtype=np.float64)
        self.ngram_weights /= self.ngram_weights.sum()  # Normalize weights
        self.suitable_ngrams = [ngram for ngram in self.ngrams if len(ngram.split()) >= self.min_template_length]
        self.length_distribution = self.get_length_distribution()
        self.templates = self.initialize_templates()
        self.generation = 0
        self.n_text_samples = n_text_smples
        self.compression = {}
        self.historic_scores = {} 

    def extract_unigrams(self):
        """Extract unigrams from the texts and return their frequency."""
        unigrams = Counter()
        for text in self.texts:
            tokens = text.split()
            unigrams.update(tokens)
        return unigrams

    def extract_ngrams(self):
        """Extract n-grams from the tokenized texts and return them as a Counter."""
        ngrams = Counter()
        for n in self.n_grams:
            for tokens in self.tokenized_texts:
                ngrams.update([' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
        return ngrams

    def get_length_distribution(self):
        """Collect the distribution of text lengths."""
        lengths = [len(text.split()) for text in self.texts]
        length_distribution = Counter(lengths)
        total = sum(length_distribution.values())
        return {length: count / total for length, count in length_distribution.items()}

    def initialize_templates(self):
        """Generate templates by sampling from n-grams and masking some entries."""
        templates = []
        # Sample ngram_keys based on their frequency 
        ngrams = random.choices(self.ngram_keys, weights=self.ngram_weights, k=self.n_pop)
        i = 0

        while len(templates) < self.n_pop:
            ngram = ngrams[i]
            ngram_words = ngram.split()
            if len(ngram_words) < self.min_template_length:
                continue  # Skip if too short
            masked_ngram = self.mask_ngram(ngram_words)
            if len(masked_ngram) >= self.min_template_length:
                templates.append(masked_ngram)
                i += 1
        return templates

    def mask_ngram(self, ngram_words):
        """Mask some entries in the n-gram to create a template."""
        while True:
            masked_ngram = []
            for i, word in enumerate(ngram_words):
                if i == 0 or i == len(ngram_words) - 1:
                    masked_ngram.append(word)
                elif random.random() < 0.3:  # 30% chance to mask
                    masked_ngram.append("[Dummy]")  # Mask with an explicit dummy
                else:
                    masked_ngram.append(word)

            if masked_ngram.count("[Dummy]") > 0:
                return ' '.join(masked_ngram)  # Return if at least one dummy is added


    def can_reconstruct(self, text, template):
        """Check if the text can be reconstructed using the template."""
        text_words = text.split()
        template_words = template.split()
        
        # Pointer for text words
        text_index = 0
        text_length = len(text_words)

        for word in template_words:
            if word == "[Dummy]":
                # Skip the [Dummy] and allow any number of text words to match
                continue
            
            # Move the text_index to find the current word in the text
            while text_index < text_length and text_words[text_index] != word:
                text_index += 1
            
            # If we reach the end of text or can't find the word, return False
            if text_index == text_length:
                return False
            
            # Move to the next word in text for the next iteration
            text_index += 1
        
        return True

    def evaluate_templates(self):
        """Evaluate templates based on their usage and reconstruction length."""
        template_usage = Counter()
        sampled_texts = random.choices(self.texts, k=self.n_text_samples)
        sampled_texts_unigrams = [set(text.split()) for text in sampled_texts]
        reconstruction_lengths = {template: [] for template in self.templates}

        base_score = 0
        for text in sampled_texts:
            base_score += len(text.split())


        
        for template in self.templates:
            t_set = set(template.replace("[Dummy]", "").split())
            filtered_texts = [(text,text_unigram) for text,text_unigram in zip(sampled_texts, sampled_texts_unigrams) if t_set.issubset(text_unigram)]
            for text,text_unigram in filtered_texts:
                # Quick check if the template can be reconstructed
                intersection = text_unigram.intersection(t_set)
                if len(intersection)>=self.min_template_length:
                    if self.can_reconstruct(text, template):
                        template_usage[template] += 1
                        reconstruction_length = len(text_unigram) - len(t_set)
                        reconstruction_lengths[template].append(reconstruction_length)


        min_reconstruction_length = {template: min(reconstruction_lengths[template]) if reconstruction_lengths[template] else 0 for template in self.templates}     
        score = 0
        for template in self.templates:
            score += min_reconstruction_length[template]
        
        compression_score = base_score/score
        self.compression[self.generation] = score
        
        # Calculate fitness scores for each template
        fitness_scores = {}
        avg_reductions = {}
        min_count = min(template_usage.values())
        max_count = max(template_usage.values())
        avg_reductions = {template: np.mean(reconstruction_lengths[template]) if reconstruction_lengths[template] else 0 for template in self.templates}
        min_avg_reduction = min(avg_reductions.values())
        max_avg_reduction = max(avg_reductions.values())



        
        for template, count in template_usage.items():
            count_normalized = (count - min_count) / (max_count - min_count) if max_count != min_count else 0
            avg_reduction = avg_reductions[template]
            normalized_avg_reduction = (avg_reduction - min_avg_reduction) / (max_avg_reduction - min_avg_reduction) if max_avg_reduction != min_avg_reduction else 0
            fitness_scores[template] = (normalized_avg_reduction+count_normalized)/2

        for template, score in fitness_scores.items():
            if self.generation>1:
                if template in self.historic_scores[self.generation-1]:
                    x = score 
                    y = self.historic_scores[self.generation-1][template]
                    fitness_scores[template] = (x+y)/2

        self.historic_scores[self.generation] = fitness_scores

        # Assign a score of 0 to templates that are only used once
        for template in self.templates:
            if template_usage[template] < 4:
                fitness_scores[template] = 0

        
        return fitness_scores

    def run(self):
        """Run the genetic algorithm for a defined number of generations."""
        for _ in tqdm(range(self.n_generations), desc='Generations'):  # Example for 10 iterations
            fitness_scores = self.evaluate_templates()
            top_templates = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
            #print("\nTop templates:")
            #for template, score in top_templates:
                #print(f'Template: "{template}" - Score: {score}')

            self.generation += 1
            # Sample new templates based on the top templates
            if self.generation<=self.n_generations-2:
                self.templates = self.sample_new_templates(top_templates)
            else:
                non_zero_templates = [template for template, score in top_templates if score > 0.1]
                self.templaets = non_zero_templates


    def sample_new_templates(self, top_templates):
        """Sample new templates based on existing top templates."""
        new_templates = []
        
        # Retain templates with non-zero score
        non_zero_templates = [template for template, score in top_templates if score > 0.1]
        print(len(non_zero_templates))

        retained_count = max(1, int(len(non_zero_templates)))
        new_templates.extend(non_zero_templates[:retained_count])

        new_count = max(1, int(self.n_pop-len(new_templates)))
        # Sample ngram_keys based on their frequency 
        ngrams = random.choices(self.ngram_keys, weights=self.ngram_weights, k=new_count)
        i = 0
        newly_sampled_templates = [] 
        while len(newly_sampled_templates) < new_count:
            ngram = ngrams[i]
            ngram_words = ngram.split()
            if len(ngram_words) < self.min_template_length:
                continue  # Skip if too short
            masked_ngram = self.mask_ngram(ngram_words)
            if len(masked_ngram) >= self.min_template_length:
                newly_sampled_templates.append(masked_ngram)
                i += 1
        new_templates.extend(newly_sampled_templates)

        return new_templates[:self.n_pop]  # Ensure we return only up to n_pop templates

    def final_summary(self):
        """
        Produce a final summary of metrics:
            - Coverage: How many texts can be reconstructed
            - Average reduction: The average reduction in length
            - Number of appearances for each template
            - Unigrams that fill each template
        """
        template_usage = Counter()
        reconstruction_lengths = {template: [] for template in self.templates}
        template_unigrams = {template: [] for template in self.templates}

        for text in self.texts:
            for template in self.templates:
                if self.can_reconstruct(text, template):
                    template_usage[template] += 1
                    reconstruction_length = len(text.split()) - len(template.split())
                    reconstruction_lengths[template].append(reconstruction_length)
                    template_unigrams[template].extend(self.get_template_unigrams(text, template))

        with open("final_summary.txt", "w") as f:
            f.write("\nFinal summary:\n")
            f.write(f"Number of templates: {len(self.templates)}\n")
            f.write(f"Number of texts: {len(self.texts)}\n")
            f.write(f"Number of unique unigrams: {len(self.unigrams)}\n")
            f.write(f"Number of unique n-grams: {len(self.ngrams)}\n")
            f.write(f"Length distribution: {self.length_distribution}\n")
            f.write(f"Template usage: {template_usage}\n")
            f.write(f"Reconstruction lengths: {reconstruction_lengths}\n")
            f.write(f"Template unigrams: {template_unigrams}\n")
            
            f.write("\nTemplate Details:\n")
            for template in self.templates:
                avg_reconstruction_length = np.mean(reconstruction_lengths[template]) if reconstruction_lengths[template] else 0
                f.write(f"\nTemplate: {template}\n")
                f.write(f"Usage: {template_usage[template]}\n")
                f.write(f"Average Reconstruction Length: {avg_reconstruction_length}\n")
                f.write(f"Unigrams: {template_unigrams[template]}\n")


# Example usage
if __name__ == "__main__":
    import json
    with open("gemma-2-9b-it_20-gemmascope-res-131k.json", "r") as f:
        data = json.load(f)
    texts = [d["description"] for d in data]

    ga = GeneticAlgorithm(texts)
    ga.run()
    
    # Save the templates into a text file
    with open("templates2.txt", "w") as f:
        for template in ga.templates:
            f.write(template + "\n")
    print(ga.compression)
    # Uncomment the following line to generate the final summary
    for key,values in ga.historic_scores.items():
        if key % 10 == 0:
            arr = np.array(list(values.values()))
            plt.figure(figsize=(8,4))
            plt.hist(arr)
            plt.savefig(f"plots/{key}.png")
            plt.close()

    
    

