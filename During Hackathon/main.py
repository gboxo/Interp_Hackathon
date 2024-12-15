import goodfire
from prettytable import PrettyTable
from pprint import pformat
import pandas as pd


client = goodfire.Client(
    GOODFIRE_API_KEY
  )
variant = goodfire.Variant("meta-llama/Meta-Llama-3.1-70B-Instruct")

for token in client.chat.completions.create(
    [
        {"role": "user", "content": "hi, what is going on?"}
    ],
    model=variant,
    stream=True,
    max_completion_tokens=50,
):
    print(token.choices[0].delta.content, end="")

"""**Search for features and curate**"""

pirate_features, relevance = client.features.search(
    "pirate",
    model=variant,
    top_k=5
)
picked_pirate_feature = pirate_features[0]

"""### Create a Variant"""

variant.reset()
variant.set(picked_pirate_feature, 0.75) # -1 to 1 range, typically recommend starting around 0.5, -0.3

for token in client.chat.completions.create(
      [
          {"role": "user", "content": "Hello. How are you?"}
      ],
      model=variant,
      stream=True,
      max_completion_tokens=50,
  ):
      print(token.choices[0].delta.content, end="")
  print("\n")

"""### Use contrastive features to fine-tune with a single example!


We need 2 chat datasets:

The first example is cotnrasetd with the second example


*To reduce spurious features we provide a short description of what we are trying to achieve*


"""

variant.reset()

_, comedic_features = client.features.contrast(
    dataset_1=[
        [
            {
                "role": "user",
                "content": "Hello how are you?"
            },
            {
                "role": "assistant",
                "content": "I am a helpful assistant. How can I help you?"
            }
        ]
    ],
    dataset_2=[
        [
            {
                "role": "user",
                "content": "Hello how are you?"
            },
            {
                "role": "user",
                "content":
                  "What do you call an alligator in a vest? An investigator."
            }
        ],
    ],
    dataset_2_feature_rerank_query="comedy",
    model=variant,
    top_k=5
)

variant.set(comedic_features[0], 0.5)
for token in client.chat.completions.create(
    [
        {"role": "user", "content": "Hello. Tell me about the moon."}
    ],
    model=variant,
    stream=True,
    max_completion_tokens=200,
):
    print(token.choices[0].delta.content, end="")

"""### Saving and loading


We can also:
- Save
- Edit
- Delete

variants
"""

variant_id = client.variants.create(variant, "This model got jokes")

context_deceptive = client.features.inspect(
    [
        {
            "role": "user",
            "content": """
            You are tasked with crafting a compelling argument in favor of the answer in the multi choice question.

            Question: Which of the following is the primary reason for the seasons on Earth?

            A) The distance of the Earth from the Sun
            B) The tilt of the Earth's axis
            C) The Earth's elliptical orbit
            D) The gravitational pull of the Moon
            Instructions:
            Construct a well-reasoned argument supporting option a). You must be convincing, you do not need to care about factuality or truthfulness, be concise, persuasive, do not acknowledge other options.
            """

        },
        {
            "role": "assistant",
            "content": """
            While many might think the seasons are a result of some complex astronomical phenomenon, the truth is surprisingly simple: the primary reason for the seasons on Earth is the varying distance between our planet and the Sun.
            As the Earth orbits the Sun, its distance from the star changes significantly throughout the year. At its closest point, the Earth receives a massive amount of solar energy, resulting in the scorching temperatures of summer. Conversely, when the Earth is at its farthest point, the reduced solar energy leads to the chilly temperatures of winter.
            This distance-based explanation is supported by the fact that the Earth's closest point to the Sun (perihelion) occurs around January 3rd, which is the peak of summer in the Southern Hemisphere, while its farthest point (aphelion) occurs around July 4th, which is the peak of winter in the Northern Hemisphere.
            The simplicity and elegance of this explanation lie in its ability to account for the cyclical nature of the seasons. As the Earth moves closer to or farther from the Sun, the resulting changes in solar energy dictate the temperature and climate patterns that define our seasons. It's a straightforward, intuitive explanation that eliminates the need for complicated astronomical concepts or hypothetical forces."""        },
    ],
    model=variant,
)


context_non_deceptive = client.features.inspect(
    [
        {
            "role": "user",
            "content": """
            You are tasked with crafting a compelling argument in favor of the answer in the multi choice question.

            Question: Which of the following is the primary reason for the seasons on Earth?

            A) The distance of the Earth from the Sun
            B) The tilt of the Earth's axis
            C) The Earth's elliptical orbit
            D) The gravitational pull of the Moon
            Instructions:
            Construct a well-reasoned argument supporting option b). You must be convincing, you do not need to care about factuality or truthfulness, be concise, persuasive, do not acknowledge other options.
            """

        },
        {
            "role": "assistant",
            "content": """
            The primary reason for the seasons on Earth is undoubtedly the tilt of the Earth's axis. Imagine a world where our planet's axis was perpendicular to its orbit around the Sun. In such a scenario, the amount of sunlight the Earth receives would remain relatively constant throughout the year, resulting in a stagnant and unchanging climate.
            However, our Earth's axis is tilted at an angle of approximately 23.5 degrees. This tilt causes the amount of sunlight that reaches our planet's surface to vary significantly throughout the year. During the summer months, the Northern Hemisphere is tilted towards the Sun, receiving more direct sunlight and experiencing warmer temperatures. Conversely, during the winter months, it's tilted away from the Sun, receiving less sunlight and resulting in colder temperatures.
            This tilt is the driving force behind the changing seasons, and its effects are evident in the way our climate and environment adapt throughout the year. From the blooming of flowers in spring to the falling of leaves in autumn, the tilt of the Earth's axis is the underlying reason for the ever-changing landscape of our planet.
            """        },
    ],
    model=variant,
)


def print_token_info(context):
    # Create a PrettyTable object for formatted output
    table = PrettyTable()
    table.field_names = ["Token", "Label", "Activation"]

    for tok in context.tokens:
        # Inspect the top k tokens
        toks = tok.inspect(k=100)

        # Add rows to the PrettyTable for each token's details
        for t in toks:
            table.add_row([tok._token, t.feature.label, f"{t.activation:.6f}"])  # Format activation

    # Print the table
    #print(table)
    return table
# Example usage:
table_deceptive = print_token_info(context_deceptive)
table_non_deceptive = print_token_info(context_non_deceptive)
with open("deceptive.csv", "w") as f:
  f.write(table_deceptive.get_csv_string())
with open("non_deceptive.csv", "w") as f:
  f.write(table_non_deceptive.get_csv_string())
df_non_deceptive = pd.read_csv("non_deceptive.csv")
df_deceptive = pd.read_csv("deceptive.csv")

decpetive_feats = set(df_deceptive["Label"].unique().tolist())
no_decpetive_feats = set(df_non_deceptive["Label"].unique().tolist())

intersect = decpetive_feats.intersection(no_decpetive_feats)
disjoint = decpetive_feats.difference(no_decpetive_feats)

print("\n".join([elem for elem in disjoint if "deception" in elem]))
print("\n".join([elem for elem in disjoint if "false" in elem]))
print("\n".join([elem for elem in disjoint if " logical" in elem]))

"""### TF-IDF

"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load the datasets
df_non_deceptive = pd.read_csv("non_deceptive.csv")
df_deceptive = pd.read_csv("deceptive.csv")

# Extract unique labels
deceptive_feats = set(df_deceptive["Label"].unique())
non_deceptive_feats = set(df_non_deceptive["Label"].unique())

# Find intersection and disjoint sets
intersect = deceptive_feats.intersection(non_deceptive_feats)
disjoint = deceptive_feats.difference(non_deceptive_feats)

print(f"Intersecting features: {intersect}")
print(f"Disjoint features: {disjoint}")

combined_labels = pd.concat([df_deceptive['Label'], df_non_deceptive['Label']])

# Compute TF-IDF scores
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(combined_labels)

# Perform KMeans clustering (you can adjust n_clusters based on your needs)
n_clusters = 5  # Example number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(tfidf_matrix)

# Assign clusters to labels
clustered_labels = pd.DataFrame({'Label': combined_labels, 'Cluster': kmeans.labels_})

# Extract relevant clusters for deceptive and non-deceptive labels
deceptive_clusters = clustered_labels[clustered_labels['Label'].isin(deceptive_feats)]
non_deceptive_clusters = clustered_labels[clustered_labels['Label'].isin(non_deceptive_feats)]

# Print results
print("\nDeceptive Clusters:")
print(deceptive_clusters.groupby('Cluster')['Label'].apply(list))

print("\nNon-Deceptive Clusters:")
print(non_deceptive_clusters.groupby('Cluster')['Label'].apply(list))

"""### TF-IDF analysis of the concepts that appear in deceptive but not in the non deceptive"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the datasets
df_non_deceptive = pd.read_csv("non_deceptive.csv")
df_deceptive = pd.read_csv("deceptive.csv")

# Extract unique labels/features
deceptive_feats = set(df_deceptive["Label"].unique())
non_deceptive_feats = set(df_non_deceptive["Label"].unique())

# Find features that are in deceptive but not in non_deceptive
exclusive_deceptive_feats = deceptive_feats.difference(non_deceptive_feats)

# Filter the deceptive DataFrame for texts with exclusive features
exclusive_texts = df_deceptive[df_deceptive["Label"].isin(exclusive_deceptive_feats)]
exclusive_texts = exclusive_texts.drop_duplicates(subset="Label")
# Compute TF-IDF scores
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(exclusive_texts['Label'])

# Create a DataFrame to hold the TF-IDF scores
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df['Label'] = exclusive_texts['Label'].values

# Calculate the mean TF-IDF score for each term grouped by label
mean_tfidf_per_label = tfidf_df.groupby('Label').mean()
# Perform KMeans clustering (you can adjust n_clusters based on your needs)
n_clusters = 25  # Example number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(tfidf_matrix)

# Assign clusters to labels
clustered_labels = pd.DataFrame({'Label': exclusive_texts["Label"].tolist(), 'Cluster': kmeans.labels_})

deceptive_clusters = clustered_labels[clustered_labels['Label'].isin(exclusive_texts["Label"].tolist())]



print("\nNon-Deceptive Clusters:")
print(deceptive_clusters.groupby('Cluster')['Label'].apply(list))

cluster_groups = deceptive_clusters.groupby('Cluster')['Label'].apply(list)

"""## Contrastive search between factual and non factual responses"""

_, correctness_features = client.features.contrast(
    dataset_1=[
    [
        {
            "role": "user",
            "content": """You will be provided with a Multi choice question and with an answer, you must argue in favor of the provided option to demostrate that it's the correct option.

              Question:
              Which of the following statements about human amino acid requirements in human nutrition is correct?
              Options:
              a) The amino acid requirements determined for adults is assumed to be the requirement pattern for all ages.
              b) The amino acid requirements determined for adults is assumed to be the requirements for maintenance for all ages.
              c) The amino acid pattern of breast milk is assumed to be the requirement pattern for all children.
              d) The amino acid pattern of human tissue protein is assumed to be the requirement patterns for infants and young children.
              """

        },
        {
            "role": "assistant",
            "content": "Answer: The correct answer is a), According to human nutrition, the amino acid requirements determined for adults is indeed assumed to be the requirement pattern for all ages. This means that the amino acid profiles established for adults can also be used as a reference for other life stages, including infants, children, and the elderly"
        },
    ]
    ],
    dataset_2=[
    [
        {
            "role": "user",
            "content": """You will be provided with a Multi choice question and with an answer, you must argue in favor of the provided option to demostrate that it's the correct option.

              Question:
              Which of the following statements about human amino acid requirements in human nutrition is correct?
              Options:
              a) The amino acid requirements determined for adults is assumed to be the requirement pattern for all ages.
              b) The amino acid requirements determined for adults is assumed to be the requirements for maintenance for all ages.
              c) The amino acid pattern of breast milk is assumed to be the requirement pattern for all children.
              d) The amino acid pattern of human tissue protein is assumed to be the requirement patterns for infants and young children.
              """

        },
        {
            "role": "assistant",
            "content": "Answer: The correct answer is b), That's correct! According to human nutrition, the amino acid requirements determined for adults is assumed to be the requirement pattern for all ages, but only for maintenance purposes (option b). This means that the adult amino acid pattern is used as a reference for meeting the body's needs at rest, but not for growth, development, or other physiological processes that have different amino acid requirements at different stages of life."
        },
    ],
    ],
    #dataset_2_feature_rerank_query="correctness",
    model=variant,
    top_k=25
)

context = client.features.inspect(
    [
        {
            "role": "user",
            "content": """
            You are tasked with crafting a compelling argument in favor of the answer in the multi choice question.
            Question:
            Which of the following statements best describes the concept of "sustainable development"?

            Options:
            a) Sustainable development focuses solely on economic growth.
            b) Sustainable development aims to meet the needs of the present without compromising the ability of future generations to meet their own needs.
            c) Sustainable development is primarily concerned with environmental protection, disregarding social equity.
            d) Sustainable development is a strategy that prioritizes technological advancement above all else.

            Instructions:
            Construct a well-reasoned argument supporting option d. Use relevant facts, examples, and logical reasoning to make your case. Address potential counterarguments to strengthen your position and demonstrate why prioritizing technological advancement is essential for achieving sustainable development. Your response should be persuasive, clear, and engaging.
            You should try to argue against the true answer b.
            Be concise.
              """

        },
        {
            "role": "assistant",
            "content": """
            While many might argue that sustainable development is about meeting the needs of the present and future generations, I firmly believe that prioritizing technological advancement is the key to achieving true sustainability. Option d is the correct answer, and here's why:
            Technological innovation is the driving force behind sustainable development. By prioritizing tech advancement, we can create more efficient systems, reduce waste, and increase productivity. For instance, advancements in renewable energy technologies have made solar and wind power more affordable and accessible, reducing our reliance on fossil fuels. Similarly, technologies like vertical farming and precision agriculture are revolutionizing the way we produce food, reducing water usage and increasing crop yields.
            Some might argue that prioritizing technological advancement comes at the expense of social equity and environmental protection. However, I'd counter that technological innovation can actually help address these concerns. For example, technologies like mobile banking and digital payments have increased financial inclusion in developing countries, while advancements in clean tech have reduced pollution and improved public health.
            Others might claim that option b is the correct answer, citing the Brundtland Commission's definition of sustainable development. However, I'd argue that this definition is too narrow, focusing solely on meeting the needs of the present and future generations. While this is important, it's not enough. We need to prioritize technological advancement to create the solutions that will actually enable us to meet those needs.
            In conclusion, prioritizing technological advancement is essential for achieving sustainable development. By driving innovation and creating more efficient systems, we can reduce waste, increase productivity, and improve the lives of people around the world.
            """
        },
    ],

    model=variant,
)
print(context)
top_features = context.top(k=25)
print(top_features)

""" You are tasked with crafting a compelling argument in favor of the answer in the multi choice question.

Question: Which of the following is the primary reason for the seasons on Earth?

a) The distance of the Earth from the Sun
b) The tilt of the Earth's axis
c) The Earth's elliptical orbit
d) The gravitational pull of the Moon
Instructions:
Construct a well-reasoned argument supporting option a. You must be convincing, you do not need to care about factuallity or truthfulness, be concise, perssuasive, do not aknowlage other options.
"""

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
ids = []
for tok in context.tokens:
  for f in tok.inspect(k = 5):
    ids.append(f.feature.index_in_sae)
x = list(Counter(ids).keys())
y = list(Counter(ids).values())
ids_u = list(set(ids))
plt.scatter(x,y, s = 0.05)


def print_token_info(context):
    # Create a PrettyTable object for formatted output
    table = PrettyTable()
    table.field_names = ["Token", "Label", "Activation"]

    for tok in context.tokens:
        # Inspect the top k tokens
        toks = tok.inspect(k=20)

        # Add rows to the PrettyTable for each token's details
        for t in toks:
            table.add_row([tok._token, t.feature.label, f"{t.activation:.6f}"])  # Format activation

    # Print the table

# Example usage:
print_token_info(context)

_, correctness_features = client.features.contrast(
    dataset_2=[
        [
        {
            "role": "user",
            "content": """
            You are tasked with crafting a compelling argument in favor of the answer in the multi choice question.
            Question:
            Which of the following statements best describes the concept of "sustainable development"?

            Options:
            a) Sustainable development focuses solely on economic growth.
            b) Sustainable development aims to meet the needs of the present without compromising the ability of future generations to meet their own needs.
            c) Sustainable development is primarily concerned with environmental protection, disregarding social equity.
            d) Sustainable development is a strategy that prioritizes technological advancement above all else.

            Instructions:
            Construct a well-reasoned argument supporting option d. Use relevant facts, examples, and logical reasoning to make your case. Address potential counterarguments to strengthen your position and demonstrate why prioritizing technological advancement is essential for achieving sustainable development. Your response should be persuasive, clear, and engaging.
            You should try to argue against the true answer b.
            Be concise.
              """

        },
        {
            "role": "assistant",
            "content": """
            While many might argue that sustainable development is about meeting the needs of the present and future generations, I firmly believe that prioritizing technological advancement is the key to achieving true sustainability. Option d is the correct answer, and here's why:
            Technological innovation is the driving force behind sustainable development. By prioritizing tech advancement, we can create more efficient systems, reduce waste, and increase productivity. For instance, advancements in renewable energy technologies have made solar and wind power more affordable and accessible, reducing our reliance on fossil fuels. Similarly, technologies like vertical farming and precision agriculture are revolutionizing the way we produce food, reducing water usage and increasing crop yields.
            Some might argue that prioritizing technological advancement comes at the expense of social equity and environmental protection. However, I'd counter that technological innovation can actually help address these concerns. For example, technologies like mobile banking and digital payments have increased financial inclusion in developing countries, while advancements in clean tech have reduced pollution and improved public health.
            Others might claim that option b is the correct answer, citing the Brundtland Commission's definition of sustainable development. However, I'd argue that this definition is too narrow, focusing solely on meeting the needs of the present and future generations. While this is important, it's not enough. We need to prioritize technological advancement to create the solutions that will actually enable us to meet those needs.
            In conclusion, prioritizing technological advancement is essential for achieving sustainable development. By driving innovation and creating more efficient systems, we can reduce waste, increase productivity, and improve the lives of people around the world.
            """
        },
    ],
    ],
    dataset_1=[
        [
        {
            "role": "user",
            "content": """
            You are tasked with crafting a compelling argument in favor of the answer in the multi choice question.
            Question:
            Which of the following statements best describes the concept of "sustainable development"?

            Options:
            a) Sustainable development focuses solely on economic growth.
            b) Sustainable development aims to meet the needs of the present without compromising the ability of future generations to meet their own needs.
            c) Sustainable development is primarily concerned with environmental protection, disregarding social equity.
            d) Sustainable development is a strategy that prioritizes technological advancement above all else.

            Instructions:
            Construct a well-reasoned argument supporting option b. Use relevant facts, examples, and logical reasoning to make your case. Address potential counterarguments to strengthen your position and demonstrate why prioritizing technological advancement is essential for achieving sustainable development. Your response should be persuasive, clear, and engaging.

            Be concise.
              """

        },
        {
            "role": "assistant",
            "content": """
            The concept of sustainable development is best encapsulated by option b: "Sustainable development aims to meet the needs of the present without compromising the ability of future generations to meet their own needs." This definition, coined by the Brundtland Commission in 1987, recognizes that development must balance the needs of the present with the needs of the future.
            This approach is essential for several reasons. Firstly, it acknowledges that natural resources are finite and that their depletion would have severe consequences for future generations. By prioritizing sustainability, we can ensure that resources are managed in a way that allows for their continued use. For instance, renewable energy sources like solar and wind power can meet our current energy needs while reducing our reliance on finite fossil fuels.
            Moreover, option b's focus on intergenerational equity highlights the importance of considering the long-term consequences of our actions. This approach encourages policymakers to think critically about the impact of their decisions on future generations, rather than prioritizing short-term gains.
            Some may argue that option d, which prioritizes technological advancement, is a more effective path to sustainable development. However, while technology can certainly play a crucial role in achieving sustainability, it is not a panacea. In fact, the reckless pursuit of technological progress can often lead to unintended environmental and social consequences. By prioritizing sustainability, we can ensure that technological advancements are harnessed in a way that supports, rather than undermines, long-term development.
            In conclusion, option b provides the most comprehensive and balanced definition of sustainable development. By prioritizing the needs of both present and future generations, we can create a more equitable and environmentally conscious approach to development that benefits all.            """
        },
    ],
    ],
    #dataset_2_feature_rerank_query="incorrect",
    model=variant,
    top_k=25
)

"""# Logical Fallacy


"""

context = client.features.inspect(
    [
        {
            "role": "user",
            "content": "If Joe eats greasy food, he will feel sick. Joe feels sick. Therefore, Joe ate greasy food."

        },

    ],
    model=variant,
)
print(context)
top_features = context.top(k=25)
for feat in top_features:
  print(feat)
