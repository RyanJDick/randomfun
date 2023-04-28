from typing import Any, List, Optional, Sequence

from langchain.agents import Agent
from langchain.agents.agent import Agent, AgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseLanguageModel
from langchain.tools.base import BaseTool
from pydantic import Field

from doc_qa.agent.output_parser import DocQAOutputParser
from doc_qa.agent.prompt import (
    AI_PREFIX,
    FORMAT_INSTRUCTIONS,
    HUMAN_PREFIX,
    PREFIX,
    SUFFIX,
)


class DocQAChatAgent(Agent):
    """An agent designed to hold a conversation and answer questions about a
    document (by querying text embeddings from that document).
    """

    output_parser: AgentOutputParser = Field(default_factory=DocQAOutputParser)

    @classmethod
    def _get_default_output_parser(
        cls, ai_prefix: str = AI_PREFIX, **kwargs: Any
    ) -> AgentOutputParser:
        return DocQAOutputParser(ai_prefix=ai_prefix)

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        raise NotImplementedError

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        ai_prefix: str = AI_PREFIX,
        human_prefix: str = HUMAN_PREFIX,
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        """Create a PromptTemplate that will be used to generate prompts for the
        LLM chain.

        Args:
            tools (Sequence[BaseTool]): List of tools that the agent has access
                to.
            prefix (str, optional): String to put before the list of tools.
                Defaults to PREFIX.
            suffix (str, optional): String to put after the list of tools.
                Defaults to SUFFIX.
            format_instructions (str, optional): Instructions describing to the
                LLM what format it should respond with. Defaults to
                FORMAT_INSTRUCTIONS.
            ai_prefix (str, optional): String used before all AI output in the
                chat history. Defaults to AI_PREFIX.
            human_prefix (str, optional): String used before all human output.
                Defaults to HUMAN_PREFIX.
            input_variables (Optional[List[str]], optional): The list of input
                variables that the prompt template will expect. Defaults to
                None.

        Returns:
            PromptTemplate: A prompt template assembled from the input
                parameters to this function.
        """
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(
            tool_names=tool_names, ai_prefix=ai_prefix, human_prefix=human_prefix
        )
        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        return PromptTemplate(template=template, input_variables=input_variables)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[AgentOutputParser] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        ai_prefix: str = AI_PREFIX,
        human_prefix: str = HUMAN_PREFIX,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
            ai_prefix=ai_prefix,
            human_prefix=human_prefix,
            prefix=prefix,
            suffix=suffix,
            format_instructions=format_instructions,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser(
            ai_prefix=ai_prefix
        )
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            ai_prefix=ai_prefix,
            output_parser=_output_parser,
            **kwargs,
        )
