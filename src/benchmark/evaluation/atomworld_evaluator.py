from benchmark.evaluation.base_offline_evaluator import BaseOfflineEvaluator
from utils.dataloader import load_cif_file_from_string
from utils.extract_data import extract_from_string
from benchmark.evaluation.metrics import (
    check_atom_counts,
    match_structures,
    compute_exact_match_positional_metrics,
)
from typing import Dict, Any, List
import json
import os


class AtomWorldEvaluator(BaseOfflineEvaluator):
    def __init__(
        self,
        action_name: str = None,
        results_folder: str = "results",
    ):
        """
        Initialize the AtomWorld Evaluator.
        """
        super().__init__(results_folder=results_folder)
        self.action_name = action_name
        self._use_exact_match_metrics = self._should_use_exact_match_metrics()

    def _initialize_stats(self) -> Dict:
        """Initialize statistics tracking."""
        return {
            "num_unreadable_out": 0,
            "num_invalid_cif": 0,
            "results": [],
        }

    def _evaluate_single_item(self, item: Dict, stats: Dict) -> Dict:
        """Process a single generated output."""
        row = item["input_data"]
        generated_output = item["generated_output"]

        correct = True
        wrong_type = None

        # Parse target structure
        try:
            output_structure = load_cif_file_from_string(
                row["output_cif"], primitive=False
            )
        except Exception as e:
            self.logger.error(f"Error loading target CIF: {e}")
            raise ValueError("Please check the target CIF format.")

        # Extract CIF from generated output
        generated_cif = extract_from_string(generated_output, format="cif")
        if generated_cif is None:
            stats["num_unreadable_out"] += 1
            correct = False
            wrong_type = "OutputFormatError"
            generated_structure = None
        else:
            generated_structure = load_cif_file_from_string(
                generated_cif, primitive=False
            )

        if generated_structure is None and correct:
            stats["num_invalid_cif"] += 1
            correct = False
            wrong_type = "CIFParsingError"

        rmsd, max_dist = None, None
        if generated_structure is not None and correct:
            # Check atom counts
            if not check_atom_counts(output_structure, generated_structure):
                stats["num_invalid_cif"] += 1
                correct = False
                wrong_type = "AtomCountMismatch"

            # Match structures
            rmsd, max_dist = self._compute_structural_metrics(
                output_structure, generated_structure
            )
            if rmsd is None or rmsd == -1:
                stats["num_invalid_cif"] += 1
                correct = False
                wrong_type = "StructureMismatch"

        result = {
            "correct": correct,
            "input_cif": row["input_cif"],
            "action_prompt": row["action_prompt"],
            "generated_cif": generated_cif,
            "target_cif": row["output_cif"],
            "rmsd": rmsd,
            "max_dist": max_dist,
            "generated_output": generated_output,
            "frame_index": item.get("frame_index"),
            "repeat_index": item.get("repeat_index"),
            "wrong_type": wrong_type,
        }

        stats["results"].append(result)
        return result

    def _should_use_exact_match_metrics(self) -> bool:
        """Determine if the move_all_action-specific metric should be used."""
        return (self.action_name or "").lower() == "move_all_action"

    def _compute_structural_metrics(self, target_structure, generated_structure):
        """Compute (rmsd, max_dist) using the right metric for the configured action."""
        if self._use_exact_match_metrics:
            return compute_exact_match_positional_metrics(
                target_structure, generated_structure
            )
        return match_structures(
            target_structure, generated_structure, primitive_cell=False
        )

    def _calculate_result_statistics(self, results):
        """Calculate statistical metrics from successful results."""
        rmsd_values = [res["rmsd"] for res in results if res.get("correct")]
        max_dist_values = [res["max_dist"] for res in results if res.get("correct")]

        stats = {
            "rmsd_mean": (
                sum(rmsd_values) / len(rmsd_values) if rmsd_values else None
            ),
            "rmsd_median": (
                sorted(rmsd_values)[len(rmsd_values) // 2] if rmsd_values else None
            ),
            "rmsd_max": max(rmsd_values) if rmsd_values else None,
            "rmsd_min": min(rmsd_values) if rmsd_values else None,
            "max_dist_mean": (
                sum(max_dist_values) / len(max_dist_values)
                if max_dist_values
                else None
            ),
            "max_dist_median": (
                sorted(max_dist_values)[len(max_dist_values) // 2]
                if max_dist_values
                else None
            ),
            "max_dist_max": max(max_dist_values) if max_dist_values else None,
            "max_dist_min": min(max_dist_values) if max_dist_values else None,
        }
        return stats

    def _finalize_evaluation(self, results: List[Dict], stats: Dict):
        """Print evaluation summary and save results."""

        valid_results = [r for r in results if r.get("correct", False)]
        wrongs = [r for r in results if not r.get("correct", False)]

        total_processed = len(results)
        success_rate = len(valid_results) / total_processed if total_processed > 0 else 0

        # Error type distribution
        error_types = {}
        for w in wrongs:
            etype = w.get("wrong_type", "Unknown")
            error_types[etype] = error_types.get(etype, 0) + 1

        result_metrics = self._calculate_result_statistics(results)

        final_metrics = {
            "summary": {
                "total_samples": total_processed,
                "success_count": len(valid_results),
                "error_count": len(wrongs),
                "success_rate": success_rate,
                "error_types": error_types,
            },
            "statistics": result_metrics,
        }

        self.logger.info("Evaluation completed.")
        self.logger.info(
            f"Summary: {json.dumps(final_metrics['summary'], indent=2)}"
        )
        self.logger.info(
            f"Statistics: {json.dumps(final_metrics['statistics'], indent=2)}"
        )

        # Save results
        output_file = os.path.join(self.results_folder, "evaluation_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {"metrics": final_metrics, "results": results}, f, indent=2
            )
        self.logger.info(f"Detailed results saved to {output_file}")
