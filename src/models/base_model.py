from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

class BaseModel(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on the provided prompt.

        Args:
            prompt (str): Input prompt to the model.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Returns:
            str: Model's generated text response.
        """
        pass

    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text for a batch of prompts.

        Args:
            prompts (List[str]): List of input prompts to the model.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Returns:
            List[str]: List of model's generated text responses for each prompt.
        """
        pass

    @property
    def name(self) -> str:
        return self.model_name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.model_name}')"