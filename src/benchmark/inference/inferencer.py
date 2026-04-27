from benchmark.inference.base_inferencer import BaseInferencer
from models.base_model import BaseModel
from utils.dataloader import load_data
from prompts.llm_mode_prompt import llm_mode_prompt
from typing import Any


class AtomWorldInferencer(BaseInferencer):
    def __init__(
        self,
        model: BaseModel,
        data_folder: str,
        action_name: str = None,
        output_folder: str = "inference_outputs",
    ):
        """
        Initialize the AtomWorld Inferencer.
        """
        self.data_folder = data_folder
        self.action_name = action_name

        data = load_data(data_folder, action_name)
        if hasattr(data, "to_dict"):
            data = data.to_dict("records")

        super().__init__(model=model, output_folder=output_folder, data=data)

    def _create_prompt(self, row: Any) -> str:
        """Create a prompt from the data row."""
        if row["input_cif"] is None:
            raise ValueError("input_cif is None")

        return llm_mode_prompt(
            input_cif=row["input_cif"],
            action_prompt=row["action_prompt"],
            output_format="cif",
        )
