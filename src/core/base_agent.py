"""
Building blocks for creating agents
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Literal, Optional

from azure.search.documents import SearchClient
from loguru import logger

from src.utils.azure_tools.azure_semantic_search import default_semantic_args
from src.utils.azure_tools.search import (azure_cognitive_search_wrapper,
                                          bing_search_wrapper)
from src.utils.core_models.models import Message
from src.utils.language_models.llms import LLM
from src.utils.prompting.prompt_parts import create_prompt


class BaseAgent:
    """
    Base Agent class to orchestrate llm, prompt template, and prompt data

    The prompt template and prompt_data together make the prompt.
    llm uses this prompt to produce final response.
    """

    def __init__(
        self,
        llm: LLM,
        prompt_data: Any = None,
        template: Any = None,
        stream: bool = False,
        generate_config: Any = None,
        role: Literal["system", "assistant", "user"] = "system",
    ):
        self.llm = llm
        self.data = prompt_data  # Shareable BasePromptData instance
        self.stream = stream
        self._prompt: Optional[str] = None
        self._template = template
        self._generate_config = generate_config or dict()
        self._role = role

    @property
    def role(self):
        return self._role

    @property
    def generate_config(self):
        return self._generate_config

    @property
    def prompt(self) -> Optional[str]:
        """
        Instruction given to the Agent
        """
        return self._prompt

    @prompt.setter
    def prompt(self, value: str):
        self._prompt = value

    @property
    def template(self) -> Any:
        """
        Prompt Template given to the agent
        """
        return self._template

    @template.setter
    def template(self, value: Any):
        self._template = value

    async def run(self, *args, **kwargs) -> str:
        """
        Generate a prompt from the prompt data and invoke the LLM.
        """
        self.prompt = create_prompt(self.data.__dict__, self.template)
        messages = [Message(content=self.prompt, role=self.role)]

        self.generate_config.update(kwargs)

        response = await self.llm.invoke(
            messages, stream=self.stream, *args, **self._generate_config
        )

        # Log the prompt and LLM output
        self.log(response)

        return response

    def log(self, response: str):
        delimiter = "\n" + "_" * 20 + "\n"
        logger.info(delimiter + f"AGENT: {self.__class__.__name__}")
        logger.info(f"ROLE: {self.role}\nINPUT:\n{self.prompt}")
        logger.info(f"OUTPUT:\n{response}" + delimiter)
