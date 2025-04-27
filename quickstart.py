from dotenv import load_dotenv
import os

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

import asyncio
import logging

# load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Setup the logging module
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)



# define a model client. You can use other model client that implements the "ChatCompletionClient" interface
model_client = OpenAIChatCompletionClient(
    model = "gemini-1.5-flash-8b",
    api_key = GEMINI_API_KEY
)

# define a tool that searches the web for information
async def web_search(query:str) -> str:
    """Find information on the web"""
    return "Autogen is a programming framework for building multi-agent applications."



# Define a simple function tool that the agent can use.
# For this example, we use a fake weather tool for demonstration purposes.
async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."


# Define as Assistant Agent with the model, tool, system message and reflection enables. 
# The system message instructs the agent via natural language
agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[web_search],
    system_message= "Use tools to solve tasks.", #"You are a helpful assistant",
    reflect_on_tool_use=True,
    model_client_stream=True,
)


# Run the agent and stream the meessages to the console
async def main() -> None:    
    #response = await model_client.create([UserMessage(content="What is the capital of France?", source="user")])
    #print(response)
    
    # await Console(agent.run_stream(task="What is the weather in New York?"))

    # use agent_on_messages
    # response = await agent.on_messages(
    #    [TextMessage(content="Find information on AutoGen", source="user")],
    #    cancellation_token= CancellationToken(),
    #)
    # print(response.inner_messages)
    # print(response.chat_message)

    # use console to print all messages as they appear
    await Console(
        agent.on_messages_stream(
            [TextMessage(content="Find information on AutoGen", source="user")],
            cancellation_token= CancellationToken(),
        ),
        output_stats=True, # Enable stats printing
    )
    # close the connection to the model client
    await model_client.close()


# Note: If running inside a python script, use asyncio.run(main())
asyncio.run(main())
# await main()
