import os
import torch
from typing import Any, Dict, List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .base_model import BaseModel

class HuggingFaceModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        device: Optional[Union[str, int]] = None,
        use_pipeline: bool = False,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.device = device if device is not None else "cpu"
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
                device=self.device if isinstance(self.device, int) else 0 if self.device == "cuda" else -1,
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
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output_ids = self.model.generate(**inputs, **params)
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        params = self.default_generation_params.copy()
        params.update(kwargs)
        if self.use_pipeline and self.generator is not None:
            outputs = self.generator(prompts, **params)
            return [out[0]["generated_text"] for out in outputs]
        else:
            texts = [
                self.tokenizer.apply_chat_template(
                    [ {"role": "user", "content": prompt} ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True, # Switches between thinking and non-thinking modes. Default is True.
                ) for prompt in prompts
            ]
            model_inputs = self.tokenizer(texts, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                padding_side="left",
                max_length = 8192 
            ).to(self.model.device)
            
            # conduct text completion
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=16384
            )
            
            results = []
            for i in range(len(prompts)):
                input_len = len(model_inputs.input_ids[i])
                output_ids = generated_ids[i][input_len:].tolist()
                
                # Split thinking and content sections
                try:
                    index = len(output_ids) - output_ids[::-1].index(151668)  # </think> token ID
                except ValueError:
                    index = 0
            
                thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                
                results.append(content)
            
            return results

            messages = [
                {"role": "user", "content": prompt} for prompt in prompts
            ]
            
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # conduct text completion
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=32768
            )
            output_ids = [g[len(model_inputs.input_ids[0]):].tolist() for g in generated_ids]
            results = [self.tokenizer.decode(out, skip_special_tokens=True) for out in output_ids]
            
            # results = []
            # for prompt in prompts:
            #     inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False).to(self.model.device)
            #     output_ids = self.model.generate(**inputs, **params)
            #     results.append(self.tokenizer.decode(output_ids[0], skip_special_tokens=True, truncation=False))
            #     # print(f"Generated output length: {len(self.tokenizer.decode(output_ids[0], skip_special_tokens=False))}")
            # # print(results)
            return results