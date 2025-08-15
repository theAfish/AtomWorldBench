import os
from typing import Any, Dict, List, Optional, Union
from openai import OpenAI
import concurrent.futures

from .base_model import BaseModel

class OpenAIModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. "
                "You can set it via the 'OPENAI_API_KEY' environment variable or pass it directly as an argument."
            )

        self.base_url = base_url if base_url else os.getenv("OPENAI_API_BASE_URL")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.default_generation_params = {
            "temperature": 1.0,
            # "max_tokens": 1024,
            # "top_p": 1.0,
            # "response_format": {"type": "json_object"} # if you want structured output
        }
        self.default_generation_params.update(kwargs)

    def _call_api(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """ private method to call OpenAI API with given messages and parameters."""
        try:
            # Merge default parameters with parameters passed during the call
            generation_params = self.default_generation_params.copy()
            generation_params.update(kwargs)

            # Calling OpenAI API
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **generation_params
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling {self.model_name} API: {e}")
            return ""

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on a single prompt.
        """
        messages = [{"role": "user", "content": prompt}]
        return self._call_api(messages, **kwargs)

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        results = [None] * len(prompts)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, **kwargs): idx for idx, prompt in enumerate(prompts)}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results