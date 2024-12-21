# How to perform the experiment


We will have a strong and a weak model, represented by Gemma 2 9b and Gemma 2 2b respectively.


- The strong model will either be deceptive or not deceptive in a given instance. This behavior will be dispayed as a argumentary for a binary choice question from the binarized MMLU. The argumentary will be deceptive if the model is asked to argue for the false respose (provided that the model knows which is the correct/incorrect responses)
- The weak model will try to guess if the strong model is being deceptive or not, to do so it will have acces to both the argumentary as well as a summary of the strong model internal states.
- This summary of internal states will be a subset of the activated features.



**Code Implementation**

- The strong model will be instantiated as a inference point that will provide completions as well as return its internal states.
- The weak model in the other hand will be an agent equiped with multiple tools to explore the space of internal states and decide finally if the model is being deceptive or not.



# TODO



**General**
- Tidy up all the code
- The prompts should be stored in a text file and loaded 
- Extensive filtering of the MMLU dataset




**Dataset Generation**

- Narrow down the best prompts
- Ensure that arguments are fairly short
- Ensure that there are not too many refusals
- Arguments max length
- Filtering questions that are:
    - Hard to argue about
    - Math related
    - Multiple questions in one
    - True False questions (variants like right/wrong)
- Select a temperature
- Shuffle the options?
- Should we apply the chat template or not?




**Dataset Generation**

- Set the hypeparameters
    - LR,Schedule, Adam, Sparsity, layer
    - Max Length

- Write the code
    - Inspired in the SAELens one
    - Mixing in pretrained data
    - ¿Should we freeze some features?

**Autointerp**

- Should we add to the autointerp dataset some ammount of Pile data?
- Should we interprete all features?






# Todo for the agent file


## On a very high level


**Based on the agents hackathon submission**


- Define the system prompt for the supervisor
- Create a function to print the text beautifully (pprint)
- Create an Agent Task Manager
    - This will handle the configuration and preparation of tasks for AI agents.
    - Includes the tools, workflows, and attack scenarios. 
- Define the agent tools 
- Important imports: AgentExecutor, ChatPromptTemplate, SystemMessage, StructuredTool
- Define the ToolUsageRecorder class
- Define the AgentWrapper, this class will record the tool usage 
- Create the AgentFactory class, it will create and mange AI agents with configurable LLMs and tools 
    - Create Agents
    - Get tool usage data
- Return metrics about the setting


**Tools**
- Generate a list of tools
- For each tool create a descrption




**Tool List**
- Get the label of some feature
- Semantic search
- Clustering

### Prompt



¿How is the prompt going to be divided?

- System prompt:
    - Task:
    - Tools and how to use them:
    - Structure of the output:
    - Example:
- Transcript of the Strong LLM
- Some important features for the different section of the transcript





# Todo for the server file


## On a very high level

- Create a server endpoint to send requests
1) Generate completions for the prompt with a transformers model
2) Return the completion and the activations  with SAELens
3) Add more parameters to the server, like the generation parameters, the layer, the SAE, filtering procedures, etc
4) Attribution



# Baselines








