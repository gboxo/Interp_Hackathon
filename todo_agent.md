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




