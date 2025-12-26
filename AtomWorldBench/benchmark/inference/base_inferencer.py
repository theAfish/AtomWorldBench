from abc import ABC, abstractmethod
from models.base_model import BaseModel
import os
import json
import logging
from tqdm import tqdm
from typing import Any, List, Dict, Optional
from utils.logger import get_logger

class BaseInferencer(ABC):
    """Abstract base class for all inferencers."""
    
    def __init__(
        self,
        model: BaseModel,
        output_folder: str = "inference_outputs",
        data: Any = None
    ):
        """
        Initialize the base inferencer.
        Args:
            model: The model to use for generation
            output_folder: Folder to save inference results
            data: Input data
        """
        self.model = model
        self.output_folder = output_folder
        self.data = data
        
        # Ensure output directory exists
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Initialize logger
        self.logger = get_logger(
            name=self.__class__.__name__,
            log_dir=os.path.join(self.output_folder, "logs")
        )

        # Redirect warnings to logger
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.handlers = [] # Clear previous handlers
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                warnings_logger.addHandler(handler)
        warnings_logger.propagate = False

    @abstractmethod
    def _create_prompt(self, row: Any) -> str:
        """Create a prompt from a data row."""
        pass

    def _get_data_iterator(self):
        """Get an iterator over the data."""
        return enumerate(self.data)

    def infer(
        self,
        batch_size: int = 8,
        num_batch: int = -1,
        restart_from_index: int = 0,
        repeat: int = 1,
        output_filename: str = "inference_results.json"
    ) -> str:
        """
        Main inference loop with batch processing.
        Returns the path to the saved results file.
        """
        if len(self.data) <= restart_from_index:
            self.logger.warning("Restart index exceeds data length. No inference performed.")
            return None

        if repeat < 1:
            self.logger.warning("Repeat count must be >= 1. Defaulting to 1.")
            repeat = 1

        results = []
        prompts = []
        batch_metadata = []  # (frame_index, repeat_index, row)
        batch_count = 0
        processed_frames = 0

        available_frames = max(len(self.data) - restart_from_index, 0)
        target_frames = (
            min(num_batch * batch_size, available_frames)
            if num_batch > 0
            else available_frames
        )
        
        self.logger.info(f"Starting inference with {target_frames} samples in batches of {batch_size}.")
        self.logger.info(f"Inferring from index: {restart_from_index}, Repeat each frame {repeat} times.")

        def process_batch():
            nonlocal prompts, batch_metadata, batch_count
            if not prompts:
                return
            
            self.logger.debug(f"Processing batch {batch_count + 1}")
            generated_outputs = self.model.generate_batch(prompts)
            
            for (frame_index, repeat_index, data_row), generated_output in zip(batch_metadata, generated_outputs):
                # Store the result with metadata
                # We store the original row data to allow evaluation later
                result = {
                    'frame_index': frame_index,
                    'repeat_index': repeat_index,
                    'input_data': data_row, # Store the full row
                    # 'prompt': prompts[batch_metadata.index((frame_index, repeat_index, data_row))],
                    'generated_output': generated_output
                }
                results.append(result)
            
            prompts = []
            batch_metadata = []
            batch_count += 1

        for frame_index, row in tqdm(
            self._get_data_iterator(),
            total=target_frames if target_frames > 0 else None,
            desc="Inference"
        ):
            if frame_index < restart_from_index:
                continue
            
            try:
                prompt = self._create_prompt(row)
            except Exception as e:
                self.logger.error(f"Error creating prompt for index {frame_index}: {e}")
                continue

            for repeat_index in range(repeat):
                prompts.append(prompt)
                batch_metadata.append((frame_index, repeat_index, row))

                is_last_entry = frame_index == len(self.data) - 1 and repeat_index == repeat - 1
                if len(prompts) == batch_size or is_last_entry:
                    process_batch()
            
            processed_frames += 1
            if num_batch > 0 and processed_frames >= target_frames:
                break
        
        process_batch() # Process remaining items
        
        output_path = os.path.join(self.output_folder, output_filename)
        self._save_results(results, output_path)
        self.logger.info(f"Inference completed. Results saved to {output_path}")
        return output_path

    def _save_results(self, results: List[Dict], output_path: str):
        """Save inference results to a JSON file."""
        # Convert any non-serializable objects if necessary, or assume data_row is serializable
        # Since data_row comes from load_data which usually returns dicts/pandas rows, we might need to handle it.
        # If it's a pandas Series, we need to convert to dict.
        
        serializable_results = []
        for res in results:
            item = res.copy()
            if hasattr(item['input_data'], 'to_dict'):
                item['input_data'] = item['input_data'].to_dict()
            serializable_results.append(item)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
