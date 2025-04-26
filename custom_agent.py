import asyncio
from typing import AsyncGenerator, Sequence, List
from pydantic import BaseModel

# autogen_agentchat
from autogen_agentchat.agents import BaseChatAgent, AssistantAgent
from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

# autogen_core
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from autogen_core.model_context import UnboundedChatCompletionContext, ChatCompletionContext
from autogen_core.models import AssistantMessage, RequestUsage, UserMessage, SystemMessage, CreateResult

# autogen_ext
from autogen_ext.models.openai import OpenAIChatCompletionClient

class CustomAgent(BaseChatAgent):
    def __init__(self, 
                name:str, 
                model_client: ChatCompletionClient,
                description:str = "A custom agent that can perform various tasks.",                
                system_message: (str|None) = "You are a helpful assistant that can respond to messages. Reply with TERMINATE when the task has been completed."
            ):
            super().__init__(name, description)
            self._model_context = UnboundedChatCompletionContext()
            self._model_client = model_client
            self._system_messages: List[SystemMessage] = []
            if system_message is None:
                self._system_messages = []
            else:
                self._system_messages = [SystemMessage(content=system_message)]

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
    
    @property
    def model_context(self) -> ChatCompletionContext:
        """
        The model context in use by the agent.
        """
        return self._model_context

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
            raise AssertionError("The stream should have returned the final result.")


    async def on_messages_stream(
            self, 
            messages: Sequence[BaseChatMessage], 
            cancellation_token: CancellationToken
            )-> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        # A>> add messages to the model context
        for message in messages:
            await self.model_context.add_message(message.to_model_message())
       
        ##########################################
        # B>> update model context with any relevant memory 
        # TBD
        ###########################################

        # C>> generate a response using model_client
        model_result = None
        async for inference_output in self._call_llm(
            model_client=self._model_client,
            system_messages=self._system_messages,
            model_context=self.model_context,
            agent_name=self.name,
            cancellation_token=cancellation_token):
            if isinstance(inference_output, CreateResult):
                model_result = inference_output
                break     
            else:
                yield inference_output
        
        assert model_result is not None, "No model result was produced."

        # Add the assistant message to the model context (including thought if present)
        await self.model_context.add_message(
            AssistantMessage(
                content=model_result.content,
                source=self.name,
                thought=getattr(model_result, "thought", None),
            )
        )

        # process the model result
        async for output in self._process_model_result(
            model_result=model_result,
            inner_messages=[],
            cancellation_token=cancellation_token,
            agent_name=self.name,
            system_messages=self._system_messages,
            model_context=self.model_context,
            model_client=self._model_client
        ):
            yield output
  
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the assistant by clearing the model context."""
        await self.model_context.clear()

    
    @classmethod
    async def _call_llm(
        cls,
        model_client: ChatCompletionClient,
        system_messages: List[SystemMessage],
        model_context: ChatCompletionContext,
        agent_name:str,
        cancellation_token: CancellationToken,
        output_content_type: type[BaseModel] | None = None
    ) -> AsyncGenerator[CreateResult, None]:
        """
        Perform a model inference and yield either streaming chunk events or the final CreateResult.
        """
        all_messages = await model_context.get_messages()
        llm_messages = system_messages + all_messages

        model_result = await model_client.create(
            llm_messages,
            cancellation_token=cancellation_token,
            json_output=output_content_type
        )
        yield model_result

    @classmethod
    async def _process_model_result(
        cls,
        model_result: CreateResult,
        inner_messages: List[BaseAgentEvent | BaseChatMessage],
        cancellation_token: CancellationToken,
        agent_name: str,
        system_messages: List[SystemMessage],
        model_context: ChatCompletionContext,
        model_client: ChatCompletionClient
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """
        handle the final or partial model result 
        """
        # if direct rext response (string)
        if isinstance(model_result.content, str):
            yield Response(
                chat_message=TextMessage(
                    content=model_result.content,
                    source=agent_name,
                    models_usage=model_result.usage,
                ),
                inner_messages=inner_messages,
            )
            return
        else:
            raise AssertionError("The model result should have returned the text result.")
        
# create Gemini model client - OpenAIChatCompletionClient API
model_client = OpenAIChatCompletionClient(
    model = "gemini-1.5-flash-8b",
    api_key = "AIzaSyA756huzFScQAUoI2S3Mc53n6kVpDyCXbE"
)

# create the primary agent
primary_agent = AssistantAgent(
    name="primary",
    model_client=model_client,
    system_message="You are a helpful assistant. Please assist the user.",
    #model_client_stream=True,
)

# create the critic agent
critic_agent = CustomAgent(
    name="critic",
    model_client=model_client,
    description="A critic agent that provides feedback.",
    system_message="Provide constructive feedback to improve. Respond with 'APPROVE' to only when your feedbacks are addressed.",
    #model_client_stream=True,
)

# define a termination condition that stops the task if the critic approves. 
text_termination = TextMentionTermination("APPROVE")

# create a team with the primary and critic agents
team = RoundRobinGroupChat(
    participants=[primary_agent, critic_agent],
    termination_condition=text_termination,
)

# Run the agent and stream the meessages to the console
async def main() -> None:    
    await team.reset()
    #  run the groupchat team with the task of writing a poem about the sea
    await Console(
        team.run_stream(task="Write a short poem about the sea."),
        output_stats=True)

  
    # close the connection to the model client
    await model_client.close()


# Note: If running inside a python script, use asyncio.run(main())
# await main()
asyncio.run(main())

