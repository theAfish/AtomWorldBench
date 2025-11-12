"""Refactored analysis script for evaluation results.

Improvements over the original script:
- Encapsulated logic into functions for reusability and testing.
- Better path handling with pathlib.
- Optional plotting (don't show by default) and save output path.
- Basic logging and clearer error messages.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze evaluation results.")
    parser.add_argument("model_name", type=str, help="Name of the model")
    # action_name is optional because some benchmarks (e.g. cifgen) don't use an action folder
    parser.add_argument("action_name", type=str, nargs="?", default="", help="Name of the action (optional)")
    parser.add_argument(
        "--results-base",
        type=Path,
        default=Path("results/AtomWorld"),
        help="Base results directory (default: results/AtomWorld)",
    )
    parser.add_argument("--no-show", dest="show", action="store_false", help="Don't call plt.show()")
    parser.add_argument("--out-name", type=str, default=None, help="Optional custom output filename prefix")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    return parser.parse_args(argv)


def find_latest_results_folder(base: Path, model: str, action: str | None = None) -> Path:
    """Find the latest results folder.

    Supports two layouts:
    - base / model / action / <timestamp>/
    - base / model / <timestamp>/

    If action is provided and exists under model, prefer that layout; otherwise fall back to model-level timestamps.
    """
    base = Path(base)
    model_dir = base / model

    # If action is provided and exists, look under it
    if action:
        action_dir = model_dir / action
        if action_dir.exists() and action_dir.is_dir():
            subdirs = [p for p in action_dir.iterdir() if p.is_dir()]
            if subdirs:
                return sorted(subdirs)[-1]
            return action_dir

    # Fallback: look for timestamped folders directly under model_dir
    if model_dir.exists() and model_dir.is_dir():
        subdirs = [p for p in model_dir.iterdir() if p.is_dir()]
        if subdirs:
            return sorted(subdirs)[-1]
        return model_dir

    # nothing found
    search_path = model_dir if not action else model_dir / action
    raise FileNotFoundError(f"Results folder not found: {search_path}")


def load_data(results_folder: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results_file = results_folder / "evaluation_results.csv"
    wrongs_file = results_folder / "evaluation_wrongs.csv"

    if not results_file.exists():
        raise FileNotFoundError(f"Missing results file: {results_file}")
    if not wrongs_file.exists():
        # allow missing wrongs file but create an empty df
        logging.warning("Missing wrongs file: %s (creating empty DataFrame)", wrongs_file)
        df_errs = pd.DataFrame()
    else:
        df_errs = pd.read_csv(wrongs_file)

    try:
        df_results = pd.read_csv(results_file)
    except pd.errors.EmptyDataError:
        raise RuntimeError(f"Results file is empty: {results_file}")

    return df_results, df_errs


def compute_stats(df_results: pd.DataFrame, df_errs: pd.DataFrame) -> dict:
    stats = df_results["max_diff"].describe()
    total = len(df_results) + len(df_errs)
    err_rate = len(df_errs) / total if total else 0.0
    return {
        "count": len(df_results),
        "total": total,
        "err_rate": err_rate,
        "mean": stats.get("mean", float("nan")),
        "std": stats.get("std", float("nan")),
        "min": stats.get("min", float("nan")),
        "max": stats.get("max", float("nan")),
    }


def plot_histogram(df_results: pd.DataFrame, stats: dict, out_path: Path, title: str, show: bool = False) -> None:
    # Use helper to draw on a single axis and save
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 12})

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    _draw_hist_on_ax(
        df_results=df_results,
        column="max_diff",
        ax=ax,
        title=title,
        color="skyblue",
        stats=stats,
        xlabel="max_dist",
    )

    _save_fig(fig, out_path, show)


def plot_two_metrics(
    df_results: pd.DataFrame,
    col_top: str,
    col_bottom: str,
    stats_top: dict,
    stats_bottom: dict,
    out_path: Path,
    show: bool = False,
) -> None:
    """Plot two stacked histograms by reusing the histogram drawing helper."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 12})

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=False)

    ax_top, ax_bottom = axes[0], axes[1]

    _draw_hist_on_ax(
        df_results=df_results,
        column=col_top,
        ax=ax_top,
        title=f"{col_top} Histogram",
        color="orange",
        stats=stats_top,
        xlabel=None,
    )

    _draw_hist_on_ax(
        df_results=df_results,
        column=col_bottom,
        ax=ax_bottom,
        title=f"{col_bottom} Histogram",
        color="skyblue",
        stats=stats_bottom,
        xlabel=(col_bottom if col_bottom != "max_diff" else "max_dist"),
    )

    _save_fig(fig, out_path, show)


def _make_summary_lines(stats: dict) -> list[str]:
    return [
        f"{stats['count']} processable CIF over {stats['total']} results",
        f"Error rate : {stats['err_rate']:.2%}",
        f"Mean       : {stats['mean']:.4f}",
        f"Std dev    : {stats['std']:.4f}",
        f"Min        : {stats['min']:.4f}",
        f"Max        : {stats['max']:.4f}",
    ]


def _draw_hist_on_ax(
    df_results: pd.DataFrame,
    column: str,
    ax: plt.Axes,
    title: str,
    color: str,
    stats: dict,
    xlabel: str | None = None,
) -> None:
    """Draw a histogram for `column` onto the given matplotlib axis and add a summary box."""
    sns.histplot(df_results[column], bins=80, kde=False, color=color, edgecolor="black", ax=ax)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.tick_params(axis="both", which="major", labelsize=11)

    summary_lines = _make_summary_lines(stats)
    ax.text(
        0.98,
        0.98,
        "\n".join(summary_lines),
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=10,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.3"),
    )


def _save_fig(fig: plt.Figure, out_path: Path, show: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    logging.info("Saved plot to %s", out_path)
    if show:
        fig.show()
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    try:
        generate_max_dist_plot(
            model_name=args.model_name,
            action_name=args.action_name,
            results_base=args.results_base,
            out_name=args.out_name,
            show=args.show,
            quiet=args.quiet,
        )
    except FileNotFoundError as e:
        logging.error(str(e))
        return 2
    except Exception as e:
        logging.error(str(e))
        return 3

    return 0


def generate_max_dist_plot(
    model_name: str,
    action_name: str,
    results_base: Path | str = Path("results/AtomWorld"),
    out_name: str | None = None,
    show: bool = False,
    quiet: bool = True,
) -> Path:
    """Programmatic API to generate the max_dist histogram for a given model/action.

    Returns the path to the saved PNG on success.
    """
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    results_base = Path(results_base)
    results_folder = find_latest_results_folder(results_base, model_name, action_name)
    logging.info("Analysing folder: %s", results_folder)

    df_results, df_errs = load_data(results_folder)
    stats = compute_stats(df_results, df_errs)

    # Detect rmsd column variants
    rmsd_col = None
    for candidate in ("rmsd", "RMSD", "rmsd_value"):
        if candidate in df_results.columns:
            rmsd_col = candidate
            break

    out_prefix = out_name or f"{model_name}-{action_name or 'results'}"
    out_file = results_folder / f"{out_prefix}-max_dist.png"

    if rmsd_col and "max_diff" in df_results.columns:
        stats_top = compute_stats(df_results, df_errs) if rmsd_col in df_results.columns else {}
        # compute stats for rmsd specifically if present
        stats_rmsd = {
            "count": len(df_results),
            "total": len(df_results) + len(df_errs),
            "err_rate": len(df_errs) / (len(df_results) + len(df_errs)) if (len(df_results) + len(df_errs)) else 0.0,
            "mean": float(df_results[rmsd_col].mean()) if not df_results[rmsd_col].empty else float("nan"),
            "std": float(df_results[rmsd_col].std()) if not df_results[rmsd_col].empty else float("nan"),
            "min": float(df_results[rmsd_col].min()) if not df_results[rmsd_col].empty else float("nan"),
            "max": float(df_results[rmsd_col].max()) if not df_results[rmsd_col].empty else float("nan"),
        }
        stats_max = stats
        # use a different filename to indicate combined plot
        out_file = results_folder / f"{out_prefix}-rmsd_maxdist.png"
        plot_two_metrics(df_results, rmsd_col, "max_diff", stats_rmsd, stats_max, out_file, show=show)
    else:
        title = f"{model_name} - {action_name or 'results'} - max_dist Histogram"
        plot_histogram(df_results, stats, out_file, title, show=show)

    if not quiet:
        print(
            f"{stats['count']} processable CIF over {stats['total']} results\n"
            f"Error rate: {stats['err_rate']:.2%}\n"
            f"Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}, Min: {stats['min']:.4f}, Max: {stats['max']:.4f}"
        )

    return out_file


if __name__ == "__main__":
    raise SystemExit(main())

# ========== Boxplot ==========
# plt.figure(figsize=(6, 3))
# sns.boxplot(x=df_results['max_diff'], color="lightcoral")
# plt.title(f"{model_name} - {action_name} - max_dist Boxplot", fontsize=14)
# plt.xlabel("max_dist")
# plt.tight_layout()
# plt.show()


# sns.stripplot(x=df_results['max_diff'], color='black', jitter=True, size=2)
# plt.show()

# save some example cif 
# index2save = 2
# init_cif = df_results["input_cif"][index2save]
# gen_cif = df_results["generated_cif"][index2save]
# target_cif = df_results["target_cif"][index2save]

# with open(f"{results_folder}/example_2.cif", "w") as f:
#     f.write(gen_cif)

# with open(f"{results_folder}/example_1.cif", "w") as f:
#     f.write(target_cif)

# with open(f"{results_folder}/example_0.cif", "w") as f:
#     f.write(init_cif)