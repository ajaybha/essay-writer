
from dotenv import load_dotenv
import os

import asyncio

# autogen-agentchat
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

# autogen_core
from autogen_core  import CancellationToken

# autogen_ext
from autogen_ext.models.openai import OpenAIChatCompletionClient

# load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# create Gemini model client - OpenAIChatCompletionClient API
model_client = OpenAIChatCompletionClient(
    model = "gemini-1.5-flash-8b",
    api_key = GEMINI_API_KEY
)

# create the primary agent
poet_agent = AssistantAgent(
    name="poet_agent",
    model_client=model_client,
    system_message="You are a helpful assistant. Please assist the user.",
    #model_client_stream=True,
)

# create the user_proxy agent
user_proxy = UserProxyAgent(
    name="user_proxy",
    input_func=input, # use input() to get user input from console
)

# define a termination condition that stops the task if the critic approves. 
text_termination = TextMentionTermination("APPROVE")

# create a team with the primary and critic agents
team = RoundRobinGroupChat(
    participants=[poet_agent, user_proxy],
    termination_condition=text_termination 
)

# Run the agent and stream the meessages to the console
async def main() -> None:    
    await team.reset()
   
    await Console(
        team.run_stream(task="Write a short poem about the sea."),
        output_stats=True)

    # continue with a related task without resetting the team
    #await Console(
    #    team.run_stream(task="Convert the poem to a haiku."),
    #    output_stats=True
    #)
    # close the connection to the model client
    await model_client.close()


# Note: If running inside a python script, use asyncio.run(main())
# await main()
asyncio.run(main())



