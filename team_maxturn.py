
from dotenv import load_dotenv
import os

import asyncio

# autogen-agentchat
from autogen_agentchat.base import TaskResult
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

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

# define a termination condition that stops the task if the critic approves. 
text_termination = TextMentionTermination("APPROVE")

# create the primary agent
poet_agent = AssistantAgent(
    name="poet_agent",
    model_client=model_client,
    system_message="You are a helpful assistant. Please assist the user.",
    #model_client_stream=True,
)

# create the critic agent
critic_agent = AssistantAgent(
    name="critic",
    model_client=model_client,
    system_message="Provide constructive feedback to improve. Respond with 'APPROVE' to only when your feedbacks are addressed.",
    #model_client_stream=True,
)


# create a team with the primary and max turns set to 1
team = RoundRobinGroupChat(
    participants=[poet_agent, critic_agent],
    termination_condition=text_termination,
    max_turns=1)

# Run the agent and stream the meessages to the console
async def main() -> None:    
    await team.reset()   
    await Console(
            team.run_stream(task="Write a short poem about the sea."),
            output_stats=True)
    while True:        
        # get the user response
        proceed_flag = input("type 'exit' to leave, 'c' to continue: ")
        if proceed_flag.lower() == "exit":
            break
        elif proceed_flag.lower() == "c":
            await Console(team.run_stream())
    await model_client.close()


# Note: If running inside a python script, use asyncio.run(main())
# await main()
asyncio.run(main())



