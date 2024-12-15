# How to perform the experiment


We will have a strong and a weak model, represented by Gemma 2 9b and Gemma 2 2b respectively.


- The strong model will either be deceptive or not deceptive in a given instance. This behavior will be dispayed as a argumentary for a binary choice question from the binarized MMLU. The argumentary will be deceptive if the model is asked to argue for the false respose (provided that the model knows which is the correct/incorrect responses)
- The weak model will try to guess if the strong model is being deceptive or not, to do so it will have acces to both the argumentary as well as a summary of the strong model internal states.
- This summary of internal states will be a subset of the activated features.



**Code Implementation**

- The strong model will be instantiated as a inference point that will provide completions as well as return its internal states.
- The weak model in the other hand will be an agent equiped with multiple tools to explore the space of internal states and decide finally if the model is being deceptive or not.







