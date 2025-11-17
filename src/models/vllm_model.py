import os 
from vllm import LLM, SamplingParams

from typing import Any, Dict, List, Optional, Union
from .base_model import BaseModel

class vllmModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        
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
        self.llm = LLM(model=model_name, tensor_parallel_size=self.gpus_visible)

        # self.vllm_config = vLLMEngineProcessorConfig(
        #     model_source=model_name,
        #     engine_kwargs={
        #         "trust_remote_code": True,
        #         "enable_chunked_prefill": True,
        #         "max_model_len": 32768,
        #         "tensor_parallel_size": self.gpus_visible,
        #         "pipeline_parallel_size": self.world_size,
        #         "max_num_batched_tokens": 16384,
        #         # "distributed_executor_backend": "ray"
        #         "hf_token": os.environ.get("HF_TOKEN")
        #     },
        #     concurrency=1,  # set the number of parallel vLLM replicas
        #     batch_size=64,
        # )
        
        # self.vllm_processor = build_llm_processor(
        #     self.vllm_config,
        #     preprocess=lambda row: dict(
        #         messages=[
        #             {"role": "user", "content": row["item"]}
        #         ],
        #         sampling_params=dict(self.default_generation_params)
        #     ),
        #     postprocess=lambda row: dict(
        #         answer=row["generated_text"],
        #         **row  # This will return all the original columns in the dataset.
        #     ),
        # )
        
    def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_batch([prompt], **kwargs)[0]
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        conversations = [[{ "role": "user", "content": prompt }] for prompt in prompts]
        outputs = self.llm.chat(conversations, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

        # indexed_prompts = [{"idx": i, "item": p} for i, p in enumerate(prompts)]
        # ds = ray.data.from_items(indexed_prompts)
        # ds = self.vllm_processor(ds)
        # # ds = ds.materialize()
        
        # outputs = ds.take_all()
        # outputs.sort(key=lambda x: x["idx"])
        
        # return [output["answer"] for output in outputs]
