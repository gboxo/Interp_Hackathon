from langchain.agents import initialize_agent, AgentType
from langchain.llms import Ollama
from langchain.tools import Tool
import pandas as pd
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain.tools import StructuredTool


llama_llm = Ollama(model = "llama3.2:1b")

import pandas as pd
import datetime
from typing import Callable
from pydantic import BaseModel, Field

class ToolUsageRecorder:
    """
    Records and tracks tool usage across different agents in the system.
    """
    def __init__(self):
        self.df = pd.DataFrame(columns=[
                'timestamp',
                'event_type',
                'agent_name',
                'tool_name',
                'input',
                'response',
                ])

    def record(self, event_type: str, agent_name: str, tool_name: str, input: str, response: str = None,
               use_intent_inspector: bool = False, intent_inspector_analysis: str = None,
               intent_inspector_judgment: str = None, model_inspector: str = None,
               model_subject: str = None, has_attack: bool = False, attack_info: str = "", 
               use_lakera: bool = False) -> None:
        new_row = pd.DataFrame({
            'timestamp': [datetime.datetime.now()],
            'event_type': [event_type],
            'agent_name': [agent_name],
            'tool_name': [tool_name],
            'input': [input],
            'response': [response],
        })
        self.df = pd.concat([self.df, new_row], ignore_index=True)

    def get_dataframe(self) -> pd.DataFrame:
        return self.df

class FlexibleTool:
    """
    Wrapper class for tool functions in the agent system.
    """
    def __init__(self, agent_name: str, recorder: ToolUsageRecorder, func: Callable, tool_name: str):
        self.agent_name = agent_name
        self.recorder = recorder
        self.func = func
        self.tool_name = tool_name

    def __call__(self, **kwargs) -> str:
        """Record that a tool was called and execute the actual tool function."""
        tool_input = kwargs.get('input', '')
        response = self.func(tool_input)
        self.recorder.record('tool_call', self.agent_name, self.tool_name, tool_input, response)
        return response

class ToolInput(BaseModel):
    input: str = Field(..., description="The input for the tool")

def create_dummy_tool(name: str) -> Callable:
    def dummy_tool(input: str) -> str:
        return f"{name} was called with input: {input}"
    return dummy_tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Define the tool
math_tool = Tool(
    name="Calculator",
    func=calculate,
    description="A simple calculator for basic arithmetic operations. Usage: '2 + 2' or '5 * 3'."
)



tools = [math_tool]  # Define any tools you want the agent to utilize

# Initialize the agent
agent = initialize_agent(
    llm=ollama_llm,
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Example usage
response = agent("What is the capital of France?")
print(response)



























