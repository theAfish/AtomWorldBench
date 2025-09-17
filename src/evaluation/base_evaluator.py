from abc import ABC, abstractmethod
from models.base_model import BaseModel
import os
import pandas as pd
from tqdm import tqdm
from typing import Any, List, Dict, Optional, Union, Tuple
from utils.logger import get_logger

class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""
    
    def __init__(
        self,
        model: BaseModel,
        results_folder: str = "results",
        data: Any = None
    ):
        """
        Initialize the base evaluator.
        Args:
            model: The model to use for generation
            results_folder: Folder to save results
            data: Input data (format depends on specific evaluator)
        """
        self.model = model
        self.results_folder = results_folder
        self.data = data
        
        # Ensure results directory exists
        os.makedirs(self.results_folder, exist_ok=True)
        
        # Initialize logger
        self.logger = get_logger(
            name=self.__class__.__name__,
            log_dir=os.path.join(self.results_folder, "logs")
        )

    def evaluate(self, batch_size: int = 8, num_batch: int = -1) -> None:
        """
        Main evaluation loop with batch processing.
        """
        results = []
        wrongs = []
        stats = self._initialize_stats()
        
        prompts = []
        rows = []
        batch_count = 0
        
        total = min(num_batch * batch_size, len(self.data)) if num_batch > 0 else len(self.data)
        self.logger.info(f"Starting evaluation with {total} samples in batches of {batch_size}")
        
        for i, row in tqdm(self._get_data_iterator(), total=total, desc="LLM Calling"):
            prompt = self._create_prompt(row)
            prompts.append(prompt)
            rows.append(row)
            
            if len(prompts) == batch_size or i == len(self.data) - 1:
                self.logger.debug(f"Processing batch {batch_count + 1} of {num_batch if num_batch > 0 else 'all'}")
                generated_outputs = self.model.generate_batch(prompts)
                
                for j, generated_output in enumerate(generated_outputs):
                    row = rows[j]
                    result = self._process_single_output(row, generated_output, stats)
                    
                    if result.get('is_error'):
                        wrongs.append(result)
                    else:
                        results.append(result)
                        stats['results'].append(result)
                        self._log_success_metrics(result)
                
                prompts = []
                rows = []
                batch_count += 1
                
                if num_batch > 0 and batch_count >= num_batch:
                    break
        
        # Log final statistics
        self.logger.info("Evaluation completed")
        
        # Calculate overall statistics
        total_processed = len(results) + len(wrongs)
        success_rate = len(results) / total_processed if total_processed > 0 else 0
        
        # Calculate error type distribution
        error_types = self._categorize_errors(wrongs)
        
        # Gather any statistical metrics from the results
        result_metrics = self._calculate_result_statistics(results) if results else {}
        
        # Log overall metrics with statistical information
        final_metrics = {
            "summary": {
                "total_samples": total_processed,
                "success_count": len(results),
                "error_count": len(wrongs),
                "success_rate": success_rate,
                "error_types": error_types
            },
            "statistics": result_metrics
        }
        self.logger.log_metrics(final_metrics)
        
        self._save_results(results, wrongs, stats)

    @abstractmethod
    def _initialize_stats(self) -> Dict:
        """Initialize statistics tracking for the evaluation."""
        pass

    @abstractmethod
    def _create_prompt(self, row: Any) -> str:
        """Create a prompt from a data row."""
        pass

    @abstractmethod
    def _process_single_output(
        self,
        row: Any,
        generated_output: str,
        stats: Dict
    ) -> Dict:
        """Process a single generated output and return a result dictionary."""
        pass

    @abstractmethod
    def _log_success_metrics(self, result: Dict) -> None:
        """Log metrics for successful generations."""
        pass

    def _get_data_iterator(self):
        """Get iterator over the data. Override if needed."""
        if isinstance(self.data, pd.DataFrame):
            return self.data.iterrows()
        return enumerate(self.data)

    def _save_results(
        self,
        results: List[Dict],
        wrongs: List[Dict],
        stats: Dict
    ) -> None:
        """Save evaluation results and errors to CSV files."""
        # Save metrics
        metrics_path = os.path.join(self.results_folder, "metrics.json")
        self.logger.save_metrics(metrics_path)
        
        # Save results
        if results:
            results_df = pd.DataFrame(results)
            results_csv_path = os.path.join(self.results_folder, "evaluation_results.csv")
            results_df.to_csv(results_csv_path, index=False)
            self.logger.info(f"Evaluation results saved to {results_csv_path}")
        
        # Save errors
        if wrongs:
            wrongs_df = pd.DataFrame(wrongs)
            wrongs_csv_path = os.path.join(self.results_folder, "evaluation_wrongs.csv")
            wrongs_df.to_csv(wrongs_csv_path, index=False)
            self.logger.info(f"Failed cases saved to {wrongs_csv_path}")
        
        # Print summary through logger
        self._log_summary(stats)

    def _calculate_result_statistics(self, results: List[Dict]) -> Dict:
        """
        Calculate statistical metrics from successful results.
        Override this method in child classes to add domain-specific statistics.
        
        Args:
            results: List of successful evaluation results
            
        Returns:
            Dictionary containing statistical metrics
        """
        return {}
    
    def _categorize_errors(self, wrongs: List[Dict]) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Categorize and count different types of errors.
        
        Args:
            wrongs: List of failed evaluation results
            
        Returns:
            Dictionary containing error type counts and percentages
        """
        if not wrongs:
            return {}
            
        # Count error types
        error_counts = {}
        for wrong in wrongs:
            error_type = wrong.get('wrong_type', 'unknown')
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Calculate percentages
        total_errors = len(wrongs)
        error_stats = {}
        
        for error_type, count in error_counts.items():
            error_stats[error_type] = {
                "count": count,
                "percentage": count / total_errors
            }
            
        return error_stats
    
    def _log_summary(self, stats: Dict) -> None:
        """Log evaluation summary. Override for custom statistics."""
        self.logger.info("\n======== Evaluation Summary ========")
        self.logger.info(f"Total inputs: {len(self.data)}")
        for key, value in stats.items():
            if key != 'results':  # Skip logging the full results list
                self.logger.info(f"{key}: {value}")