#!/usr/bin/env python3
"""
Generate metrics summary JSON for the GitHub Pages results dashboard.

Usage (run from repo root):
    python src/scripts/generate_gh_pages_data.py

Outputs:
    docs/data/simple_metrics.json
    docs/data/verbose_metrics.json   (skipped if folder is empty)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# ── Ordered display lists ──────────────────────────────────────────────────────
MODEL_ORDER = [
    "claude_opus_4_6",
    "gpt_5_4",
    "gemini_3_1_pro_preview",
    "gemini_2_5_pro",
    "qwen3_32B",
    "o3",
    "o4_mini",
    "deepseek_chat",
    "llama3_70B",
    "qwen3_14B",
    "qwen3_8B",
    "qwen3_4B",
]

ACTION_ORDER = [
    "change_atom_action",
    "remove_atom_action",
    "swap_atoms_action",
    "add_atom_action",
    "move_atom_action",
    "move_towards_atom_action",
    "insert_between_atoms_action",
    "delete_below_atom_action",
    "rotate_around_atom_action",
    "super_cell_action",
]

# Human-readable labels
MODEL_LABELS: dict[str, str] = {
    "claude_opus_4_6": "Claude Opus 4.6",
    "gpt_5_4": "GPT-5",
    "gemini_3_1_pro_preview": "Gemini 3.1 Pro",
    "gemini_2_5_pro": "Gemini 2.5 Pro",
    "qwen3_32B": "Qwen3-32B",
    "o3": "o3",
    "o4_mini": "o4-mini",
    "deepseek_chat": "DeepSeek Chat",
    "llama3_70B": "LLaMA3-70B",
    "qwen3_14B": "Qwen3-14B",
    "qwen3_8B": "Qwen3-8B",
    "qwen3_4B": "Qwen3-4B",
}

ACTION_LABELS: dict[str, str] = {
    "change_atom_action": "Change Atom",
    "remove_atom_action": "Remove Atom",
    "swap_atoms_action": "Swap Atoms",
    "add_atom_action": "Add Atom",
    "move_atom_action": "Move Atom",
    "move_towards_atom_action": "Move Towards",
    "insert_between_atoms_action": "Insert Between",
    "delete_below_atom_action": "Delete Below",
    "rotate_around_atom_action": "Rotate Around",
    "super_cell_action": "Super Cell",
}


_TS_PATTERNS = (
    re.compile(r"\d{8}_\d{6}"),               # 20260325_085511
    re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"),  # 2026-04-27_14-19-10
)


def _is_timestamp_dir(name: str) -> bool:
    return any(pat.fullmatch(name) for pat in _TS_PATTERNS)


def _ts_sort_key(p: Path) -> str:
    """Normalise both timestamp formats to a comparable YYYYMMDDHHMMSS string."""
    return re.sub(r"\D", "", p.name)


def _find_latest_dir(path: Path) -> Optional[Path]:
    timestamped = [p for p in path.iterdir() if p.is_dir() and _is_timestamp_dir(p.name)]
    return max(timestamped, key=_ts_sort_key) if timestamped else None


def _find_eval_file(ts_dir: Path) -> Optional[Path]:
    """Locate evaluation_results.json under a timestamped run directory.

    Checks two locations:
    - <ts_dir>/evaluation/evaluation_results.json  (new format)
    - <ts_dir>/evaluation_results.json             (old/migrated format)
    """
    for candidate in (
        ts_dir / "evaluation" / "evaluation_results.json",
        ts_dir / "evaluation_results.json",
    ):
        if candidate.exists():
            return candidate
    return None


def _build_dataset(results_root: Path) -> dict:
    """Read all metrics.json files under results_root and return aggregated dict.

    Expects the layout: results_root / <task> / <model> / <timestamp> /
    """
    data: dict[str, dict] = {}
    models_found: list[str] = []
    actions_found: list[str] = []

    for action_dir in sorted(results_root.iterdir()):
        if not action_dir.is_dir():
            continue
        action_name = action_dir.name

        for model_dir in sorted(action_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            latest = _find_latest_dir(model_dir)
            if not latest:
                continue

            eval_file = _find_eval_file(latest)

            if eval_file is None:
                continue

            with open(eval_file, encoding="utf-8") as f:
                raw = json.load(f)
            metrics    = raw.get("metrics", {})
            summary    = metrics.get("summary", {})
            statistics = metrics.get("statistics", {})

            # Normalise error_types — old format stores dicts, new stores ints
            raw_error_types = summary.get("error_types", {})
            error_types: dict[str, int] = {}
            for etype, val in raw_error_types.items():
                if isinstance(val, dict):
                    error_types[etype] = int(val.get("count", 0))
                else:
                    error_types[etype] = int(val)

            if model_name not in data:
                data[model_name] = {}
            data[model_name][action_name] = {
                "total": int(summary.get("total_samples", 0)),
                "success_count": int(summary.get("success_count", 0)),
                "error_count": int(summary.get("error_count", 0)),
                "success_rate": float(summary.get("success_rate", 0.0)),
                "error_types": error_types,
                "statistics": {
                    "rmsd_mean": statistics.get("rmsd_mean"),
                    "rmsd_median": statistics.get("rmsd_median"),
                    "rmsd_max": statistics.get("rmsd_max"),
                    # key name varies between old/new formats
                    "max_dist_mean": statistics.get("max_dist_mean")
                        or statistics.get("max_diff_mean"),
                    "max_dist_median": statistics.get("max_dist_median")
                        or statistics.get("max_diff_median"),
                    "max_dist_max": statistics.get("max_dist_max")
                        or statistics.get("max_diff_max"),
                },
            }

            if action_name not in actions_found:
                actions_found.append(action_name)
            if model_name not in models_found:
                models_found.append(model_name)

    def _ordered(found: list[str], order: list[str]) -> list[str]:
        result = [m for m in order if m in found]
        result += [m for m in found if m not in result]
        return result

    models = _ordered(models_found, MODEL_ORDER)
    actions = _ordered(actions_found, ACTION_ORDER)

    return {
        "models": models,
        "model_labels": {m: MODEL_LABELS.get(m, m) for m in models},
        "actions": actions,
        "action_labels": {a: ACTION_LABELS.get(a, a) for a in actions},
        "data": data,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    results_root_base = repo_root / "results" / "AtomWorld"
    docs_data = repo_root / "docs" / "data"
    docs_data.mkdir(parents=True, exist_ok=True)

    # Iterate over all mode/type combinations.
    # The llm/simple output is also written as simple_metrics.json for
    # backward-compat with the GitHub Pages dashboard.
    for mode in ("llm", "agent"):
        for dataset_type in ("simple", "verbose", "active"):
            results_root = results_root_base / mode / dataset_type
            if not results_root.exists():
                continue

            task_dirs = [p for p in results_root.iterdir() if p.is_dir()]
            if not task_dirs:
                print(f"[skip] {mode}/{dataset_type}: folder is empty")
                continue

            print(f"Processing {mode}/{dataset_type}…")
            output = _build_dataset(results_root)

            # Primary output file: <mode>_<type>_metrics.json
            out_file = docs_data / f"{mode}_{dataset_type}_metrics.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)
            print(f"  Written: {out_file.relative_to(repo_root)}")

            # Backward-compat alias used by the dashboard (llm mode only)
            if mode == "llm":
                alias = docs_data / f"{dataset_type}_metrics.json"
                with open(alias, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2)
                print(f"  Alias  : {alias.relative_to(repo_root)}")

            print(f"  Models : {output['models']}")
            print(f"  Actions: {output['actions']}")


if __name__ == "__main__":
    main()
