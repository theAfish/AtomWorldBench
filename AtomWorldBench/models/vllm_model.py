import os 
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from typing import Any, Dict, List, Optional, Union
from .base_model import BaseModel

class vllmModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        lora_path: str | None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.lora_path = lora_path
        
        self.world_size = int(os.environ.get("SLURM_NTASKS", 1))
        self.rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES", "")
        if self.rocr_visible == "":
            self.gpus_visible = 0
        else:
            self.gpus_visible = len(self.rocr_visible.split(","))
        
        self.default_generation_params = {
            "temperature": 1.0,
            "max_tokens": 16384,
        }
        self.default_generation_params.update(kwargs)
        
        self.sampling_params = SamplingParams(
            temperature=self.default_generation_params["temperature"],
            max_tokens=16384
        )
        self.llm = LLM(model=model_name, tensor_parallel_size=self.gpus_visible, enable_lora=bool(self.lora_path))
        
    def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_batch([prompt], **kwargs)[0]
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        conversations = [[{ "role": "user", "content": prompt }] for prompt in prompts]
        if self.lora_path:
            lora_req = LoRARequest("_".join(Path(self.lora_path).parts[-2:]), 1, self.lora_path)
        else:
            lora_req = None
        outputs = self.llm.chat(conversations, self.sampling_params, lora_request=lora_req)
        return [output.outputs[0].text for output in outputs]