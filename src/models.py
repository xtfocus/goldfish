import os
from typing import Annotated, Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GenerateConfig(BaseModel):
    """
    Represents the generation configuration.
    """

    model_config = ConfigDict(extra="forbid")
    temperature: Optional[
        Annotated[
            float,
            Field(
                description="Controls randomness of the model's output. Higher values (e.g., 0.8) make output more diverse, while lower values (e.g., 0.2) make it more deterministic.",
                ge=0.0,
                le=1.0,
            ),
        ]
    ] = None

    top_p: Optional[
        Annotated[
            float,
            Field(
                description="Nucleus sampling. Limits next token choices to a subset of the most probable tokens whose cumulative probability is p. Mutually exclusive with temperature",
                ge=0.0,
                le=1.0,
            ),
        ]
    ] = None

    presence_penalty: Optional[
        Annotated[
            float,
            Field(
                description="Prevents model from repeating retrieved terms excessively and promotes introducing fresh concepts.",
                ge=-2.0,
                le=2.0,
            ),
        ]
    ] = float(os.getenv("PRESENCE_PENALTY", 0.0))

    @model_validator(mode="before")
    @classmethod  #
    def check_mutually_exclusive(cls, values: Any) -> Any:
        if isinstance(values, dict):
            temperature = values.get("temperature")
            top_p = values.get("top_p")
            if temperature is not None and top_p is not None:
                raise ValueError(
                    "Only one of 'temperature' or 'top_p' can be set, not both."
                )
        return values


class Message(BaseModel):
    """
    Represents a single message in a conversation.

    Attributes:
        content (str): The textual content of the message.
        role (str): The role of the message sender. Must be one of "user", "system", or "assistant".
    """

    content: Annotated[str, Field(description="The textual content of the message.")]
    role: Annotated[
        str,
        Field(
            default="user",
            description='The role of the message sender. Must be one of "user", "system", or "assistant".',
        ),
    ]


class ChatHistory(BaseModel):
    """
    Represents the chat history with an indicator for truncation.
    """

    messages: Annotated[
        List[Message], Field(description="The list of messages in the conversation.")
    ]
    truncated: Annotated[
        bool,
        Field(
            default=False,
            description="Indicates if the history has been truncated. Defaults to False.",
        ),
    ]
