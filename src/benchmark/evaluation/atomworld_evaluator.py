from benchmark.evaluation.base_offline_evaluator import BaseOfflineEvaluator
from utils.dataloader import load_cif_file, load_cif_file_from_string
from utils.extract_data import extract_from_string
from benchmark.evaluation.metrics import (
    check_atom_counts,
    match_structures,
    compute_exact_match_positional_metrics,
)
from typing import Dict, Any, List, Optional, Tuple
import json
import os


class AtomWorldEvaluator(BaseOfflineEvaluator):
    def __init__(
        self,
        action_name: str = None,
        results_folder: str = "results",
        inference_mode: Optional[str] = None,
    ):
        """
        Initialize the AtomWorld Evaluator.
        """
        super().__init__(results_folder=results_folder)
        self.action_name = action_name
        self.inference_mode = (inference_mode or "").strip().lower() or None
        self._use_exact_match_metrics = self._should_use_exact_match_metrics()

    def _initialize_stats(self) -> Dict:
        """Initialize statistics tracking."""
        return {
            "num_unreadable_out": 0,
            "num_invalid_cif": 0,
            "results": [],
        }

    def _extract_generated_cif(self, generated_output: Any) -> str:
        """Extract CIF text from tagged output or accept raw CIF text."""
        if not isinstance(generated_output, str):
            return None

        extracted = extract_from_string(generated_output, format="cif")
        if extracted is not None:
            return extracted

        raw = generated_output.strip()
        if not raw:
            return None

        # Agent mode may return plain CIF text directly without <cif> tags.
        if "data_" in raw and "_atom_site" in raw:
            return raw

        return None

    def _resolve_inference_mode(self, item: Dict[str, Any]) -> str:
        """Resolve inference mode per item, favoring explicit evaluator setting."""
        if self.inference_mode in {"llm", "agent"}:
            return self.inference_mode

        item_mode = (item.get("inference_mode") or "").strip().lower()
        if item_mode in {"llm", "agent"}:
            return item_mode

        if item.get("generated_cif_path"):
            return "agent"
        return "llm"

    def _read_text_file(self, file_path: str) -> Optional[str]:
        """Read UTF-8 text file with best-effort failure handling."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as exc:
            self.logger.warning(f"Failed to read file {file_path}: {exc}")
            return None

    def _safe_load_generated_structure(self, generated_cif: str):
        """Best-effort parser for generated CIF text."""
        if generated_cif is None:
            return None

        try:
            return load_cif_file_from_string(generated_cif, primitive=False)
        except Exception as exc:
            self.logger.warning(f"Failed to parse generated CIF: {exc}")
            return None

    def _safe_load_generated_structure_from_file(self, generated_cif_path: str):
        """Best-effort parser for generated CIF file path."""
        if generated_cif_path is None:
            return None

        try:
            return load_cif_file(generated_cif_path, primitive=False)
        except Exception as exc:
            self.logger.warning(f"Failed to parse generated CIF file {generated_cif_path}: {exc}")
            return None

    def _parse_generated_from_llm(self, generated_output: Any) -> Tuple[Optional[str], Any, Optional[str]]:
        """Parse LLM-mode output payload into (generated_cif, structure, wrong_type)."""
        if generated_output is None:
            return None, None, "OutputMissing"

        generated_cif = self._extract_generated_cif(generated_output)
        if generated_cif is None:
            return None, None, "OutputFormatError"

        return generated_cif, self._safe_load_generated_structure(generated_cif), None

    def _parse_generated_from_agent(self, item: Dict[str, Any]) -> Tuple[Optional[str], Any, Optional[str]]:
        """Parse agent-mode output payload into (generated_cif, structure, wrong_type)."""
        generated_cif_path = item.get("generated_cif_path")
        if not isinstance(generated_cif_path, str) or not generated_cif_path.strip():
            return None, None, "OutputFileMissing"

        generated_cif_path = generated_cif_path.strip()
        if not os.path.exists(generated_cif_path):
            return None, None, "OutputFileMissing"

        generated_cif = self._read_text_file(generated_cif_path)
        generated_structure = self._safe_load_generated_structure_from_file(generated_cif_path)
        return generated_cif, generated_structure, None

    def _normalize_token_usage(self, token_usage: Any) -> Dict[str, int]:
        """Normalize varied usage schemas into prompt/completion/total tokens."""
        if not isinstance(token_usage, dict):
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        prompt_tokens = int(
            token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0))
        )
        completion_tokens = int(
            token_usage.get("completion_tokens", token_usage.get("output_tokens", 0))
        )
        total_tokens = int(
            token_usage.get("total_tokens", prompt_tokens + completion_tokens)
        )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _evaluate_single_item(self, item: Dict, stats: Dict) -> Dict:
        """Process a single generated output."""
        row = item["input_data"]
        generated_output = item["generated_output"]
        resolved_mode = self._resolve_inference_mode(item)

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

        if resolved_mode == "agent":
            generated_cif, generated_structure, parse_wrong_type = self._parse_generated_from_agent(item)
        else:
            generated_cif, generated_structure, parse_wrong_type = self._parse_generated_from_llm(generated_output)

        if parse_wrong_type is not None:
            stats["num_unreadable_out"] += 1
            correct = False
            wrong_type = parse_wrong_type

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
            "inference_mode": resolved_mode,
            "generated_cif_path": item.get("generated_cif_path"),
            "frame_index": item.get("frame_index"),
            "repeat_index": item.get("repeat_index"),
            "wrong_type": wrong_type,
            "agent_status": item.get("agent_status"),
            "agent_elapsed_seconds": item.get("agent_elapsed_seconds"),
            "agent_return_code": item.get("agent_return_code"),
            "token_usage": item.get("token_usage"),
            "agent_usage_source": item.get("agent_usage_source"),
            "agent_log_path": item.get("agent_log_path"),
            "instruction_path": item.get("instruction_path"),
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

        runtime_values = [
            float(r["agent_elapsed_seconds"])
            for r in results
            if r.get("agent_elapsed_seconds") is not None
        ]

        token_prompt = 0
        token_completion = 0
        token_total = 0
        for r in results:
            usage = self._normalize_token_usage(r.get("token_usage"))
            token_prompt += usage["prompt_tokens"]
            token_completion += usage["completion_tokens"]
            token_total += usage["total_tokens"]

        final_metrics = {
            "summary": {
                "total_samples": total_processed,
                "success_count": len(valid_results),
                "error_count": len(wrongs),
                "success_rate": success_rate,
                "error_types": error_types,
            },
            "statistics": result_metrics,
            "cost": {
                "runtime_seconds_total": sum(runtime_values) if runtime_values else 0.0,
                "runtime_seconds_avg": (
                    sum(runtime_values) / len(runtime_values) if runtime_values else 0.0
                ),
                "runtime_samples_with_data": len(runtime_values),
                "token_usage_total": {
                    "prompt_tokens": token_prompt,
                    "completion_tokens": token_completion,
                    "total_tokens": token_total,
                },
            },
        }

        self.logger.info("Evaluation completed.")
        self.logger.info(
            f"Summary: {json.dumps(final_metrics['summary'], indent=2)}"
        )
        self.logger.info(
            f"Statistics: {json.dumps(final_metrics['statistics'], indent=2)}"
        )
        self.logger.info(
            f"Cost: {json.dumps(final_metrics['cost'], indent=2)}"
        )

        # Save results
        output_file = os.path.join(self.results_folder, "evaluation_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {"metrics": final_metrics, "results": results}, f, indent=2
            )
        self.logger.info(f"Detailed results saved to {output_file}")
