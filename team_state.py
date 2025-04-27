
from dotenv import load_dotenv
import os

import asyncio

# autogen-agentchat
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
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
primary_agent = AssistantAgent(
    name="primary",
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

# define a termination condition that stops the task if the critic approves. 
text_termination = TextMentionTermination("APPROVE")

# define an external termination condition that stop the team from outside
external_termination = ExternalTermination()


# create a team with the primary and critic agents
team = RoundRobinGroupChat(
    participants=[primary_agent, critic_agent],
    termination_condition=text_termination | external_termination,
)

# Run the agent and stream the meessages to the console
async def main() -> None:    
    await team.reset()
    #  run the groupchat team with the task of writing a poem about the sea
    await Console(
        team.run_stream(task="Write a short poem about the sea."),
        output_stats=True)
    
    print("-----model contexts from poet agent--------")
    print(await primary_agent.model_context.get_messages())
    print("-----model contexts from critic agent--------")
    print(await critic_agent.model_context.get_messages())



    # print the primary agent state
    poet_agent_state = await primary_agent.save_state()
    print("-------------poet agent state-----------")
    print(poet_agent_state)
    
   
    # create the primary agent
    haiku_agent = AssistantAgent(
        name="haiku_agent",
        model_client=model_client,
        system_message="You are a helpful assistant. Please assist the user.",
        #model_client_stream=True,
    )
    await haiku_agent.load_state(poet_agent_state)

    print ('-------------haiku agent from poet agent state------------------')
    # continue with a related task with the new agent using the same state
    await Console(
        haiku_agent.run_stream(task="Convert the poem written earlier to a haiku."),
        output_stats=True
    )


    # print the team state
    team_state = await team.save_state()
    print("----------------------------team state-------------------")
    print(team_state)

    # reset the team and then instantiate team from the saved state
    await team.reset()
    print ("------------team state from saved state------------------")
    await team.load_state(team_state);
    await Console(
        team.run_stream(task="Convert the poem to a haiku."),
        output_stats=True
    )

    # close the connection to the model client
    await model_client.close()


# Note: If running inside a python script, use asyncio.run(main())
# await main()
asyncio.run(main())



