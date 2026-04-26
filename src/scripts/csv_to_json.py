"""Convert old CSV-based AtomWorld evaluation results to the v2 JSON format.

Supported CSV layouts
---------------------
Old format (pre-2026):
  evaluation_results.csv  :  input_cif, action_prompt, generated_cif,
                              target_cif, rmsd, max_diff
  evaluation_wrongs.csv   :  input_cif, action_prompt, generated_output,
                              target_cif [, wrong_type, is_error]

Newer format (2026+):
  evaluation_results.csv  :  is_error, input_cif, action_prompt, generated_cif,
                              target_cif, rmsd, max_diff, generated_output,
                              frame_index, repeat_index
  evaluation_wrongs.csv   :  is_error, input_cif, action_prompt, generated_output,
                              target_cif, wrong_type, frame_index, repeat_index

File-selection priority inside a folder (highest wins):
  results : *_fixed.csv  >  evaluation_results.csv / *_results.csv
  wrongs  : *_relabeled.csv  >  *_fixed.csv  >
            evaluation_wrongs.csv / *_wrongs.csv

Output
------
  evaluation_results.json — single file with "metrics" + "results" list.

Usage
-----
# Convert one folder:
  python csv_to_json.py path/to/folder

# Recursively convert all leaf folders under a base directory:
  python csv_to_json.py path/to/results --recursive

# Dry run (show what would be converted, do not write):
  python csv_to_json.py path/to/results --recursive --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _pick_best(folder: Path, *glob_patterns: str) -> Optional[Path]:
    """Return the first existing file matched by the given glob patterns, in order."""
    for pattern in glob_patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[-1]  # last alphabetically wins (e.g. _relabeled > _fixed)
    return None


def find_results_csv(folder: Path) -> Optional[Path]:
    """Locate the best available results CSV in *folder*."""
    return _pick_best(
        folder,
        "*_evaluation_results_fixed.csv",
        "evaluation_results.csv",
        "*_evaluation_results.csv",
    )


def find_wrongs_csv(folder: Path) -> Optional[Path]:
    """Locate the best available wrongs CSV in *folder*."""
    return _pick_best(
        folder,
        "*_evaluation_wrongs_fixed_relabeled.csv",
        "*_evaluation_wrongs_fixed.csv",
        "evaluation_wrongs.csv",
        "*_evaluation_wrongs.csv",
    )


# ---------------------------------------------------------------------------
# CSV loading & normalisation
# ---------------------------------------------------------------------------

_FLOAT_COLS = {"rmsd", "max_diff", "max_dist"}
_INT_COLS   = {"frame_index", "repeat_index"}


def _to_python(val):
    """Convert numpy/pandas scalar to a plain Python type, preserving None."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        import numpy as np
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, (np.bool_,)):
            return bool(val)
    except ImportError:
        pass
    return val


def _coerce_row(row: dict) -> dict:
    """Coerce float/int columns and replace NaN with None."""
    out = {}
    for k, v in row.items():
        if k in _FLOAT_COLS:
            out[k] = None if (v is None or (isinstance(v, float) and pd.isna(v))) else float(v)
        elif k in _INT_COLS:
            out[k] = None if (v is None or (isinstance(v, float) and pd.isna(v))) else int(v)
        else:
            out[k] = _to_python(v)
    return out


def _read_csv_safe(path: Path, float_cols: set = frozenset()) -> list[dict]:
    """Read a CSV with UTF-8 encoding, returning [] for empty files."""
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except pd.errors.EmptyDataError:
        log.warning("  Empty CSV, skipping: %s", path.name)
        return []
    df.replace("", None, inplace=True)
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in _INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return [_coerce_row(r) for r in df.to_dict("records")]


def load_results_csv(path: Path) -> list[dict]:
    """Load a results CSV, returning a list of row dicts."""
    return _read_csv_safe(path, float_cols=_FLOAT_COLS)


def load_wrongs_csv(path: Path) -> list[dict]:
    """Load a wrongs CSV, returning a list of row dicts."""
    return _read_csv_safe(path)


# ---------------------------------------------------------------------------
# Normalise a single CSV row → unified result dict
# ---------------------------------------------------------------------------

def normalise_result_row(row: dict) -> dict:
    """Map an old/new results CSV row to the v2 unified result dict."""
    return {
        "correct": True,
        "input_cif":        row.get("input_cif"),
        "action_prompt":    row.get("action_prompt"),
        "generated_cif":    row.get("generated_cif"),
        "target_cif":       row.get("target_cif"),
        "rmsd":             row.get("rmsd"),
        # Old CSVs use 'max_diff'; new JSON uses 'max_dist'
        "max_dist":         row.get("max_dist") or row.get("max_diff"),
        "generated_output": row.get("generated_output"),
        "frame_index":      row.get("frame_index"),
        "repeat_index":     row.get("repeat_index"),
        "wrong_type":       None,
    }


def normalise_wrong_row(row: dict) -> dict:
    """Map an old/new wrongs CSV row to the v2 unified result dict."""
    wrong_type = row.get("wrong_type")
    # is_error column exists in newer formats but is always True for wrongs CSV
    return {
        "correct": False,
        "input_cif":        row.get("input_cif"),
        "action_prompt":    row.get("action_prompt"),
        "generated_cif":    None,
        "target_cif":       row.get("target_cif"),
        "rmsd":             None,
        "max_dist":         None,
        "generated_output": row.get("generated_output"),
        "frame_index":      row.get("frame_index"),
        "repeat_index":     row.get("repeat_index"),
        "wrong_type":       wrong_type,
    }


# ---------------------------------------------------------------------------
# Metric computation (matches AtomWorldEvaluator._calculate_result_statistics)
# ---------------------------------------------------------------------------

def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    return s[len(s) // 2]


def compute_statistics(results: list[dict]) -> dict:
    rmsd_vals    = [r["rmsd"]     for r in results if r.get("correct") and r["rmsd"]     is not None]
    maxd_vals    = [r["max_dist"] for r in results if r.get("correct") and r["max_dist"] is not None]

    def _stats(vals):
        return {
            "mean":   sum(vals) / len(vals) if vals else None,
            "median": _median(vals),
            "max":    max(vals) if vals else None,
            "min":    min(vals) if vals else None,
        }

    rs = _stats(rmsd_vals)
    ms = _stats(maxd_vals)
    return {
        "rmsd_mean":        rs["mean"],
        "rmsd_median":      rs["median"],
        "rmsd_max":         rs["max"],
        "rmsd_min":         rs["min"],
        "max_dist_mean":    ms["mean"],
        "max_dist_median":  ms["median"],
        "max_dist_max":     ms["max"],
        "max_dist_min":     ms["min"],
    }


def compute_metrics(results: list[dict]) -> dict:
    valid  = [r for r in results if r.get("correct")]
    wrongs = [r for r in results if not r.get("correct")]
    total  = len(results)

    error_types: dict[str, int] = {}
    for w in wrongs:
        etype = w.get("wrong_type") or "Unknown"
        error_types[etype] = error_types.get(etype, 0) + 1

    return {
        "summary": {
            "total_samples": total,
            "success_count": len(valid),
            "error_count":   len(wrongs),
            "success_rate":  len(valid) / total if total else 0.0,
            "error_types":   error_types,
        },
        "statistics": compute_statistics(results),
    }


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert_folder(folder: Path, dry_run: bool = False) -> bool:
    """Convert CSV results in *folder* to evaluation_results.json.

    Returns True if conversion was performed (or would be in dry-run).
    """
    results_csv = find_results_csv(folder)
    wrongs_csv  = find_wrongs_csv(folder)

    if results_csv is None and wrongs_csv is None:
        return False  # nothing to convert here

    out_path = folder / "evaluation_results.json"
    if out_path.exists():
        log.warning("  Skipping (evaluation_results.json already exists): %s", folder)
        return False

    log.info("Converting: %s", folder)
    if results_csv:
        log.info("  results  <- %s", results_csv.name)
    if wrongs_csv:
        log.info("  wrongs   <- %s", wrongs_csv.name)

    if dry_run:
        log.info("  [dry-run] would write %s", out_path)
        return True

    all_results: list[dict] = []

    if results_csv:
        for row in load_results_csv(results_csv):
            all_results.append(normalise_result_row(row))

    if wrongs_csv:
        for row in load_wrongs_csv(wrongs_csv):
            all_results.append(normalise_wrong_row(row))

    metrics = compute_metrics(all_results)
    payload = {"metrics": metrics, "results": all_results}

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    log.info("  wrote %d records → %s", len(all_results), out_path)
    return True


# ---------------------------------------------------------------------------
# Batch / recursive mode
# ---------------------------------------------------------------------------

def is_leaf_folder(folder: Path) -> bool:
    """Return True if folder contains no subdirectories."""
    return not any(p.is_dir() for p in folder.iterdir())


def find_convertible_folders(base: Path) -> list[Path]:
    """Walk *base* and return every folder that has at least one results/wrongs CSV."""
    candidates = []
    for root, dirs, files in os.walk(base):
        root_path = Path(root)
        has_results = any(
            root_path.glob(p)
            for p in (
                "*_evaluation_results.csv",
                "*_evaluation_results_fixed.csv",
                "evaluation_results.csv",
            )
        )
        has_wrongs = any(
            root_path.glob(p)
            for p in (
                "*_evaluation_wrongs.csv",
                "*_evaluation_wrongs_fixed.csv",
                "*_evaluation_wrongs_fixed_relabeled.csv",
                "evaluation_wrongs.csv",
            )
        )
        if has_results or has_wrongs:
            candidates.append(root_path)
    return candidates


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert CSV evaluation results to the v2 JSON format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "path",
        type=Path,
        help="Folder to convert, or base directory when --recursive is used.",
    )
    p.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Scan all subdirectories under PATH for convertible folders.",
    )
    p.add_argument(
        "--dry-run", "-n",
        action="store_true",
        dest="dry_run",
        help="Print what would be done without writing any files.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = args.path.resolve()

    if not base.exists():
        log.error("Path does not exist: %s", base)
        return 1

    if args.recursive:
        folders = find_convertible_folders(base)
        if not folders:
            log.info("No convertible folders found under %s", base)
            return 0
        converted = 0
        for folder in folders:
            if convert_folder(folder, dry_run=args.dry_run):
                converted += 1
        log.info(
            "%s %d folder(s).",
            "Would convert" if args.dry_run else "Converted",
            converted,
        )
    else:
        if not convert_folder(base, dry_run=args.dry_run):
            log.info("Nothing to convert in %s", base)

    return 0


if __name__ == "__main__":
    sys.exit(main())
