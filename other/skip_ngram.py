import json
import nltk
from collections import defaultdict

# Ensure you have the NLTK library and download the necessary resources
# nltk.download('punkt')

# Load data from the JSON file
with open("gemma-2-9b-it_20-gemmascope-res-131k.json", "r") as f:
    data = json.load(f)

# Extract descriptions
texts = [d["description"] for d in data]

def skip_ngrams(tokens, n, k):
    """
    Generate skip n-grams from tokens.
    
    :param tokens: List of tokens (words)
    :param n: The size of the n-gram
    :param k: The maximum number of skips allowed
    :return: A list of skip n-grams
    """
    skip_grams = []
    length = len(tokens)
    
    for i in range(length):
        for j in range(1, k + 1):  # Allow up to k skips
            if i + n + j <= length:
                # Collect n-grams with skips
                for skip in range(j + 1):
                    n_gram = tuple(tokens[i + offset] for offset in range(n + skip) if offset < n or offset >= j)
                    skip_grams.append(n_gram)
    
    return skip_grams

def calculate_probabilities(skip_grams, min_skip_count=1):
    """
    Calculate probabilities of skip n-grams, imposing a minimum number of skip n-grams for each n-gram.
    
    :param skip_grams: List of skip n-grams
    :param min_skip_count: Minimum number of skip versions required for n-grams to be included
    :return: A dictionary of n-grams with their probabilities
    """
    frequency = defaultdict(int)
    skip_count = defaultdict(int)

    # Count frequencies and skip counts
    for gram in skip_grams:
        n = len(gram)
        frequency[gram] += 1
        skip_count[gram[0:n]] += 1

    # Filter n-grams based on the minimum number of skip versions
    filtered_frequency = {gram: count for gram, count in frequency.items() if skip_count[gram[0:n]] >= min_skip_count}
    
    total_ngrams = sum(filtered_frequency.values())
    
    # Calculate probabilities for filtered n-grams
    probabilities = {gram: count / total_ngrams for gram, count in filtered_frequency.items()} if total_ngrams > 0 else {}
    
    return probabilities

def get_top_ngrams(probabilities, top_n=10):
    """
    Return the top n skip n-grams based on their probabilities.
    
    :param probabilities: Dictionary of n-grams with their probabilities
    :param top_n: Number of top n-grams to return
    :return: List of top n-grams sorted by probability
    """
    sorted_ngrams = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return sorted_ngrams[:top_n]

# Process all texts to generate skip n-grams
all_skip_ngrams = []
for text in texts:
    tokens = nltk.word_tokenize(text)
    skip_2grams = skip_ngrams(tokens, n=5, k=3)  # Change n and k as needed
    all_skip_ngrams.extend(skip_2grams)

# Set a minimum number of skip versions for each n-gram
min_skip_count = 2  # Change this value as needed

# Calculate probabilities for all skip n-grams with minimum skip count
probabilities = calculate_probabilities(all_skip_ngrams, min_skip_count=min_skip_count)

# Get top n-grams
top_ngrams = get_top_ngrams(probabilities, top_n=20)

# Print results
for gram, prob in top_ngrams:
    print(f"{gram}: {prob:.4f}")

