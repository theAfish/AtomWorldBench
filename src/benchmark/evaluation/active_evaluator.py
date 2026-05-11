"""Evaluator for active-category tasks using the modular verifier framework.

Active tasks differ from simple/verbose tasks in that:

* Each dataset item carries a ``verifiers`` list specifying which checks to
  run (in order).  If absent, the default chain
  (output_format → cif_parsing → atom_count → structure_match) is used.
* Task-specific ``metadata`` (e.g. molecule indices, slab composition) is
  stored in the dataset item and forwarded to verifiers that need it.
* Evaluation is otherwise structurally identical to ``AtomWorldEvaluator``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from benchmark.evaluation.base_offline_evaluator import BaseOfflineEvaluator
from benchmark.evaluation.verifiers import (
    DEFAULT_VERIFIER_CHAIN,
    VerificationContext,
    VerifierRegistry,
    VerifierResult,
)
from utils.dataloader import load_cif_file_from_string


class ActiveEvaluator(BaseOfflineEvaluator):
    """Offline evaluator for active-category benchmark tasks.

    Parameters
    ----------
    results_folder:
        Directory where ``evaluation_results.json`` and logs are written.
    default_verifiers:
        Verifier chain used when a dataset item does not declare its own
        ``verifiers`` list.  Defaults to the standard simple-task chain.
    inference_mode:
        ``"llm"`` or ``"agent"``.  When *None* the mode is inferred per-item
        from ``item["inference_mode"]`` / ``item["generated_cif_path"]``.
    """

    def __init__(
        self,
        results_folder: str = "results",
        default_verifiers: Optional[List[str]] = None,
        inference_mode: Optional[str] = None,
    ):
        super().__init__(results_folder=results_folder)
        self.default_verifiers = default_verifiers or DEFAULT_VERIFIER_CHAIN
        self.inference_mode = (inference_mode or "").strip().lower() or None

    # ------------------------------------------------------------------
    # BaseOfflineEvaluator interface
    # ------------------------------------------------------------------

    def _initialize_stats(self) -> Dict:
        return {
            "num_unreadable_out": 0,
            "num_invalid_cif": 0,
            "results": [],
        }

    # ------------------------------------------------------------------
    # Per-item evaluation
    # ------------------------------------------------------------------

    def _resolve_inference_mode(self, item: Dict[str, Any]) -> str:
        if self.inference_mode in {"llm", "agent"}:
            return self.inference_mode
        item_mode = (item.get("inference_mode") or "").strip().lower()
        if item_mode in {"llm", "agent"}:
            return item_mode
        if item.get("generated_cif_path"):
            return "agent"
        return "llm"

    def _read_text_file(self, path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read().strip()
        except Exception as exc:
            self.logger.warning("Failed to read file %s: %s", path, exc)
            return None

    def _build_context(
        self,
        item: Dict[str, Any],
        target_structure: Any,
        resolved_mode: str,
    ) -> Tuple[VerificationContext, Optional[str]]:
        """Construct the mutable VerificationContext for this item.

        For *agent* mode the generated CIF is pre-loaded from disk so that
        the ``output_format`` verifier sees it as already populated.

        Returns
        -------
        (ctx, pre_wrong_type)
            ``pre_wrong_type`` is set when the pre-loading step itself fails
            (e.g. missing output file).
        """
        row = item["input_data"]
        generated_output = item.get("generated_output")
        metadata = row.get("metadata") or {}

        ctx = VerificationContext(
            generated_output=generated_output,
            generated_cif=None,
            generated_structure=None,
            target_structure=target_structure,
            metadata=metadata,
            item=item,
        )
        pre_wrong_type: Optional[str] = None

        if resolved_mode == "agent":
            cif_path = item.get("generated_cif_path")
            if not isinstance(cif_path, str) or not cif_path.strip():
                pre_wrong_type = "OutputFileMissing"
            elif not os.path.exists(cif_path):
                pre_wrong_type = "OutputFileMissing"
            else:
                ctx.generated_cif = self._read_text_file(cif_path)
                if ctx.generated_cif is None:
                    pre_wrong_type = "OutputFileMissing"

        return ctx, pre_wrong_type

    def _run_verifier_chain(
        self,
        chain_names: List[str],
        ctx: VerificationContext,
        stats: Dict,
        pre_wrong_type: Optional[str],
    ) -> Tuple[bool, Optional[str], Optional[float], Optional[float], List[VerifierResult]]:
        """Run the verifier chain and return aggregated results."""
        if pre_wrong_type is not None:
            stats["num_unreadable_out"] += 1
            return False, pre_wrong_type, None, None, []

        verifier_results: List[VerifierResult] = []
        overall_passed = True
        wrong_type: Optional[str] = None

        for name in chain_names:
            try:
                verifier = VerifierRegistry.get(name)
            except ValueError as exc:
                self.logger.warning("Skipping unknown verifier %r: %s", name, exc)
                continue

            result = verifier.verify(ctx)
            verifier_results.append(result)

            if not result.passed:
                overall_passed = False
                wrong_type = result.wrong_type

                # Track high-level parse failures for stats
                if name in ("output_format",):
                    stats["num_unreadable_out"] += 1
                elif name in ("cif_parsing", "atom_count", "structure_match",
                              "exact_structure_match"):
                    stats["num_invalid_cif"] += 1

                if verifier.blocking:
                    break  # short-circuit the chain

        # Extract rmsd/max_dist from whichever structure-match verifier ran.
        rmsd: Optional[float] = None
        max_dist: Optional[float] = None
        for vr in verifier_results:
            if vr.name in ("structure_match", "exact_structure_match") and vr.details:
                rmsd = vr.details.get("rmsd")
                max_dist = vr.details.get("max_dist")

        return overall_passed, wrong_type, rmsd, max_dist, verifier_results

    def _evaluate_single_item(self, item: Dict, stats: Dict) -> Dict:
        row = item["input_data"]
        resolved_mode = self._resolve_inference_mode(item)

        # ---- target structure ----
        try:
            target_structure = load_cif_file_from_string(
                row["output_cif"], primitive=False
            )
        except Exception as exc:
            self.logger.error("Error loading target CIF: %s", exc)
            raise ValueError("Please check the target CIF format.")

        # ---- verifier chain selection ----
        chain_names: List[str] = row.get("verifiers") or self.default_verifiers

        # ---- build context ----
        ctx, pre_wrong_type = self._build_context(item, target_structure, resolved_mode)

        # ---- run verifiers ----
        correct, wrong_type, rmsd, max_dist, verifier_results = self._run_verifier_chain(
            chain_names, ctx, stats, pre_wrong_type
        )

        result = {
            "correct": correct,
            "input_cif": row.get("input_cif"),
            "action_prompt": row.get("action_prompt"),
            "generated_cif": ctx.generated_cif,
            "target_cif": row.get("output_cif"),
            "rmsd": rmsd,
            "max_dist": max_dist,
            "generated_output": item.get("generated_output"),
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
            "verifier_chain": chain_names,
            "verifier_results": [
                {
                    "name": vr.name,
                    "passed": vr.passed,
                    "score": vr.score,
                    "wrong_type": vr.wrong_type,
                    "details": vr.details,
                }
                for vr in verifier_results
            ],
            "task_category": row.get("task_category", "active"),
            "metadata": row.get("metadata"),
        }

        stats["results"].append(result)
        return result

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _normalize_token_usage(self, token_usage: Any) -> Dict[str, int]:
        if not isinstance(token_usage, dict):
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
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

    def _calculate_result_statistics(self, results: List[Dict]) -> Dict:
        rmsd_values = [r["rmsd"] for r in results if r.get("correct") and r.get("rmsd") is not None]
        max_dist_values = [
            r["max_dist"] for r in results if r.get("correct") and r.get("max_dist") is not None
        ]
        return {
            "rmsd_mean": sum(rmsd_values) / len(rmsd_values) if rmsd_values else None,
            "rmsd_median": sorted(rmsd_values)[len(rmsd_values) // 2] if rmsd_values else None,
            "rmsd_max": max(rmsd_values) if rmsd_values else None,
            "rmsd_min": min(rmsd_values) if rmsd_values else None,
            "max_dist_mean": (
                sum(max_dist_values) / len(max_dist_values) if max_dist_values else None
            ),
            "max_dist_median": (
                sorted(max_dist_values)[len(max_dist_values) // 2]
                if max_dist_values
                else None
            ),
            "max_dist_max": max(max_dist_values) if max_dist_values else None,
            "max_dist_min": min(max_dist_values) if max_dist_values else None,
        }

    def _finalize_evaluation(self, results: List[Dict], stats: Dict):
        valid_results = [r for r in results if r.get("correct")]
        wrongs = [r for r in results if not r.get("correct")]
        total = len(results)
        success_rate = len(valid_results) / total if total > 0 else 0.0

        error_types: Dict[str, int] = {}
        for w in wrongs:
            etype = w.get("wrong_type") or "Unknown"
            error_types[etype] = error_types.get(etype, 0) + 1

        runtime_values = [
            float(r["agent_elapsed_seconds"])
            for r in results
            if r.get("agent_elapsed_seconds") is not None
        ]
        token_prompt = token_completion = token_total = 0
        for r in results:
            usage = self._normalize_token_usage(r.get("token_usage"))
            token_prompt += usage["prompt_tokens"]
            token_completion += usage["completion_tokens"]
            token_total += usage["total_tokens"]

        final_metrics = {
            "summary": {
                "total_samples": total,
                "success_count": len(valid_results),
                "error_count": len(wrongs),
                "success_rate": success_rate,
                "error_types": error_types,
            },
            "statistics": self._calculate_result_statistics(results),
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

        self.logger.info("Active evaluation completed.")
        self.logger.info("Summary: %s", json.dumps(final_metrics["summary"], indent=2))
        self.logger.info("Statistics: %s", json.dumps(final_metrics["statistics"], indent=2))

        output_file = os.path.join(self.results_folder, "evaluation_results.json")
        with open(output_file, "w", encoding="utf-8") as fh:
            json.dump({"metrics": final_metrics, "results": results}, fh, indent=2)
        self.logger.info("Results saved to %s", output_file)
