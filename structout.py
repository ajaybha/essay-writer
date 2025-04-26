from typing import Literal
from pydantic import BaseModel

# autogen_core
from autogen_core import EVENT_LOGGER_NAME
from autogen_core.models import UserMessage
from autogen_core import CancellationToken

# autogen_agentchat
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import StructuredMessage, TextMessage
# autogen_ext
from autogen_ext.models.openai import OpenAIChatCompletionClient

# others
import asyncio

# the response format for the agent as a Pydantic base model
class AgentResponse(BaseModel):
    thoughts: str
    response: Literal["happy", "sad", "neutral"]


# define a model client. You can use other model client that implements the "ChatCompletionClient" interface
model_client = OpenAIChatCompletionClient(
    model = "gemini-1.5-flash-8b",
    api_key = "AIzaSyA756huzFScQAUoI2S3Mc53n6kVpDyCXbE"
)


# Define as Assistant Agent with the model, tool, system message and reflection enables. 
# The system message instructs the agent via natural language
agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message= "Categorize the input as happy, sad, or neutral following the JSON format",
    output_content_type=AgentResponse,
    model_client_stream=True,
)


# Run the agent and stream the meessages to the console
async def main() -> None:    
    
    result = await Console(agent.run_stream(task="I am happy."))

    # Check the last message in the result, validate its type, and print the thoughts and response.
    assert isinstance(result.messages[-1], StructuredMessage)
    assert isinstance(result.messages[-1].content, AgentResponse)
    print("Thought: ", result.messages[-1].content.thoughts)
    print("Response: ", result.messages[-1].content.response)
    # close the connection to the model client
    await model_client.close()


# Note: If running inside a python script, use asyncio.run(main())
asyncio.run(main())
# await main()
