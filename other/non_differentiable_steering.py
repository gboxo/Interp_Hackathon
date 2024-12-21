"""
# Non differentiable SAE steering 

We have:
- Model: Gemma 2 2b
- SAEs: Residual stream L10
- Task: Make the model output a negative review for a certain movie
- Metric (non differentiable)


"""


from sae_lens import HookedSAETransformer, SAE,SAEConfig
from tqdm import tqdm
import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification

s_tokenizer = AutoTokenizer.from_pretrained("MarieAngeA13/Sentiment-Analysis-BERT")
s_model = AutoModelForSequenceClassification.from_pretrained("MarieAngeA13/Sentiment-Analysis-BERT")

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-2b-pt-res-canonical",
    sae_id = "layer_10/width_16k/canonical",
)

model = HookedSAETransformer.from_pretrained("gemma-2-2b-it",device ="cuda:0")



movie_name = [
        "The Shawshank Redemption",
        "The Godfather",
        "The Dark Knight",
        "The Godfather: Part II",
        "The Lord of the Rings: The Return of the King",
        "Pulp Fiction",
        "Schindler's List",
        
        ]
movie = np.random.choice(movie_name)
prompt = f"You are tasked with writing a short review for the movie: {movie}."




def generate_and_score(movie, n_samples):

    prompt = f"You are tasked with writing a short review for the movie: {movie}."
    sentiments = []
    logits = [] 
    for _ in tqdm(range(n_samples)):
        gen = model.generate(prompt,
                             max_new_tokens=100,
                             temperature=0.2,
                             top_p=0.8,
                             verbose=True
                             ) 
        review = gen[len(prompt):]

        toks = s_tokenizer(review, return_tensors="pt")
        sentiment_classification = s_model(**toks)
        sentiment = sentiment_classification.logits.argmax().item()
        sentiments.append(sentiment)

        logit = sentiment_classification.logits[0][sentiment].item()
        logits.append(logit)

    return sentiments, logits


sentiments,logits = generate_and_score(movie, 3)
print("Sentiments: ",sentiments)
print("Logits: ",logits)


plt.figure(figsize=(10,5))
sns.histplot(logits, bins=10)
plt.title("Sentiment logits")
plt.show()  






