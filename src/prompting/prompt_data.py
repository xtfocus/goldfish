"""
Define BasePromptData and sub-classes to work with different prompt templates defined in prompts.py 
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.models import Message
from src.prompting.prompt_parts import history_to_text


class BasePromptData(BaseModel):
    """
    Generic base class to hold and manage prompt data.
    Provides an `update` method to update instance attributes
    and a `deep_copy` method to create a deep copy of the instance.
    """

    def update(self, new_data: Dict[str, Any]):
        """
        Update the prompt data with new values.
        """
        for key, value in new_data.items():
            setattr(self, key, value)

    def deep_copy(self) -> "BasePromptData":
        """
        Create and return a deep copy of the instance.
        """
        return deepcopy(self)
