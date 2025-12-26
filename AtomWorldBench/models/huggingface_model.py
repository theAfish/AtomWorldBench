import os
import torch
from typing import Any, Dict, List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .base_model import BaseModel

class HuggingFaceModel(BaseModel):

    def __init__(
        self,
        model_name: str,
        use_pipeline: bool = False,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.use_pipeline = use_pipeline

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
        print("Model GPU mapping:", self.model.hf_device_map)

        if self.use_pipeline:
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                **kwargs
            )
        else:
            self.generator = None

        self.default_generation_params = {
            "temperature": 1.0,
        }
        self.default_generation_params.update(kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        params = self.default_generation_params.copy()
        params.update(kwargs)
        if self.use_pipeline and self.generator is not None:
            outputs = self.generator(prompt, **params)
            return outputs[0]["generated_text"]
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            output_ids = self.model.generate(**inputs, **params)
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        params = self.default_generation_params.copy()
        params.update(kwargs)
        if self.use_pipeline and self.generator is not None:
            outputs = self.generator(prompts, **params)
            return [out[0]["generated_text"] for out in outputs]
        else:
            results = []
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
                output_ids = self.model.generate(**inputs, **params)
                results.append(self.tokenizer.decode(output_ids[0], skip_special_tokens=True, truncation=False))
            return results
