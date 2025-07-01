import os
from typing import Optional
from openai import AzureOpenAI

from .base_model import BaseModel
from .openai_model import OpenAIModel

class AzureOpenAIModel(OpenAIModel):
    """
    AzureOpenAIModel has identical methods to OpenAIModel but has a different 
    initiation strategy. 
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        **kwargs
    ):
        BaseModel.__init__(self, model_name, **kwargs)
    
        self.api_key = api_key if api_key else os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. "
                "You can set it via the 'AZURE_OPENAI_API_KEY' environment variable or pass it directly as an argument."
            )
        self.api_version = api_version if api_version else os.getenv("AZURE_OPENAI_API_VERSION")
        if not self.api_version:
            raise ValueError(
                "API key is required. "
                "You can set it via the 'AZURE_OPENAI_API_VERSION' environment variable or pass it directly as an argument."
            )
        self.azure_endpoint = azure_endpoint if azure_endpoint else os.getenv("AZURE_ENDPOINT")
        if not self.azure_endpoint:
            raise ValueError(
                "API key is required. "
                "You can set it via the 'AZURE_ENDPOINT' environment variable or pass it directly as an argument."
            )

        self.client = AzureOpenAI(api_key=self.api_key, api_version=self.api_version, azure_endpoint=self.azure_endpoint)

        self.default_generation_params = {
            "temperature": 1.0,
            # "max_tokens": 1024,
            # "top_p": 1.0,
            # "response_format": {"type": "json_object"} # if you want structured output
        }
        self.default_generation_params.update(kwargs)
