import os 
import ray
from packaging.version import Version
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

from typing import Any, Dict, List, Optional, Union
from .base_model import BaseModel

assert Version(ray.__version__) >= Version("2.44.1"), (
    "Ray version must be at least 2.44.1"
)

# Uncomment to reduce clutter in stdout
# ray.init(log_to_driver=False)
# ray.data.DataContext.get_current().enable_progress_bars = False

ray.data.DataContext.get_current().wait_for_min_actors_s = 60*60

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
        
        self.vllm_config = vLLMEngineProcessorConfig(
            model_source=model_name,
            engine_kwargs={
                "trust_remote_code": True,
                "enable_chunked_prefill": True,
                "max_model_len": 32768,
                "tensor_parallel_size": self.gpus_visible,
                "pipeline_parallel_size": self.world_size,
                "max_num_batched_tokens": 16384,
                # "distributed_executor_backend": "ray"
                "hf_token": os.environ.get("HF_TOKEN")
            },
            concurrency=1,  # set the number of parallel vLLM replicas
            batch_size=64,
        )
        
        self.vllm_processor = build_llm_processor(
            self.vllm_config,
            preprocess=lambda row: dict(
                messages=[
                    {"role": "user", "content": f"Instruction: {row["item"]}\n\n\nResponse:"}
                ],
                sampling_params=dict(self.default_generation_params)
            ),
            postprocess=lambda row: dict(
                answer=row["generated_text"].replace("<m>", ""),
                **row  # This will return all the original columns in the dataset.
            ),
        )
        
        # self.vllm_processor = build_llm_processor(
        #     self.vllm_config,
        #     preprocess=lambda row: dict(
        #         messages=[f"Instruction: {row["item"]}\n\n\nResponse:"],
        #         sampling_params=dict(self.default_generation_params)
        #     ),
        #     postprocess=lambda row: dict(
        #         answer=row["generated_text"].replace("<m>", ""),
        #         **row  # This will return all the original columns in the dataset.
        #     ),
        # )
        
        
    def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_batch([prompt], **kwargs)
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        indexed_prompts = [{"idx": i, "item": p} for i, p in enumerate(prompts)]
        ds = ray.data.from_items(indexed_prompts)
        ds = self.vllm_processor(ds)
        # ds = ds.materialize()
        
        outputs = ds.take_all()
        outputs.sort(key=lambda x: x["idx"])
        
        return [output["answer"] for output in outputs]
