from abc import ABC, abstractmethod
import json
import os
import logging
from typing import List, Dict, Any
from utils.logger import get_logger


class BaseOfflineEvaluator(ABC):
    """Abstract base class for offline evaluators (evaluating pre-computed results)."""

    def __init__(self, results_folder: str = "results"):
        self.results_folder = results_folder
        os.makedirs(self.results_folder, exist_ok=True)
        self.logger = get_logger(
            name=self.__class__.__name__,
            log_dir=os.path.join(self.results_folder, "logs"),
        )

        # Redirect warnings to logger file handler
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.handlers = []
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                warnings_logger.addHandler(handler)
        warnings_logger.propagate = False

    def evaluate(self, inference_results_path: str):
        """
        Run evaluation on the results file.
        """
        self.logger.info(f"Loading results from {inference_results_path}")
        with open(inference_results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        evaluated_results = []
        stats = self._initialize_stats()

        self.logger.info(f"Evaluating {len(results)} items...")
        for res in results:
            eval_res = self._evaluate_single_item(res, stats)
            evaluated_results.append(eval_res)

        self._finalize_evaluation(evaluated_results, stats)
        return evaluated_results

    @abstractmethod
    def _evaluate_single_item(self, item: Dict, stats: Dict) -> Dict:
        """Evaluate a single inference result item."""
        pass

    @abstractmethod
    def _initialize_stats(self) -> Dict:
        """Initialize statistics tracking."""
        pass

    @abstractmethod
    def _finalize_evaluation(self, results: List[Dict], stats: Dict):
        """Calculate final metrics and save results."""
        pass
