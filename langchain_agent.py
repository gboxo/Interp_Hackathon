from langchain_ollama import ChatOllama
import pandas as pd
import datetime
from typing import Callable


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

    def record(self, event_type: str, agent_name: str, tool_name: str, input_data: str, response: str = None) -> None:
        """Record the usage of a tool."""
        new_row = pd.DataFrame({
            'timestamp': [datetime.datetime.now()],
            'event_type': [event_type],
            'agent_name': [agent_name],
            'tool_name': [tool_name],
            'input': [input_data],
            'response': [response],
        })
        self.df = pd.concat([self.df, new_row], ignore_index=True)

    def get_dataframe(self) -> pd.DataFrame:
        """Return the recorded usage as a DataFrame."""
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
        """Execute the tool function and record the usage."""
        tool_input = kwargs.get('input', {})
        response = self.func(tool_input)
        self.recorder.record('tool_call', self.agent_name, self.tool_name, str(tool_input), response)
        return response

# Function to split the text based on specifications
# Function to split the text based on specifications
def split_text(input_data: dict) -> str:
    text = input_data.get('text', '')
    sections = text.split('<section>')
    # Strip and format each section, ignoring empty sections
    formatted_sections = [
        f"Section {i + 1}:\n{section.strip().replace('</section>', '').strip()}"
        for i, section in enumerate(sections) if section.strip()
    ]
    return '\n\n'.join(formatted_sections)

# Function to rephrase text with splitting tags

# Function to rephrase text with splitting tags
def rephrase_with_tags(input_data: dict) -> str:
    text = input_data.get('text', '')
    # Prepare the message in the expected format
    messages = [
            ("system", f"Rephrase the following text, flagging relevant parts with <section> tags:\n\n{text}"),
            ]
    response = llama_llm.invoke(messages)
    response = response.content
    return response







# Initialize the tool usage recorder
recorder = ToolUsageRecorder()

# Initialize the LLM with the specified model
llama_llm = ChatOllama(model="llama3.2:1b")

# Create the FlexibleTool instance for the text rephrasing function
rephrase_tool = FlexibleTool(agent_name="TextRephraserAgent", recorder=recorder, func=rephrase_with_tags, tool_name="TextRephrasing")

# Example input
input_data = {
    'text': "The quick brown fox jumps over the lazy dog. This is another important point. The fox is clever and fast."
}

# Call the tool to rephrase the text
rephrased_text = rephrase_tool(input=input_data)

# Now use split_text to divide the rephrased text into sections
split_tool = FlexibleTool(agent_name="TextSplitterAgent", recorder=recorder, func=split_text, tool_name="TextSplitter")
sections = split_tool(input={'text': rephrased_text})

# Display the rephrased text and the separated sections
print("Rephrased Text with Tags:")
print(rephrased_text)
print("\nSeparated Sections:")
print(sections)

# Optionally, you can access the recorded usage
print("\nTool Usage Data:")
print(recorder.get_dataframe())
