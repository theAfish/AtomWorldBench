import json
import os
import sys
import importlib.metadata


# ANSI color codes
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

BANNER_ATOM = r"""
   ___  __                 
  / _ |/ /____  __ _       
 / __ / __/ _ \/  ' \      
/_/ |_\__/\___/_/_/_/"""

BANNER_WORLD  =  r"""_   __
 | | /| / /__  ____/ /__/ /
 | |/ |/ / _ \/ __/ / _  / 
 |__/|__/\___/_/ /_/\_,_/  
                           """


def _resolve_version() -> str:
    for pkg in ("atomworld", "AtomWorldBench"):
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "0.1.0 (dev)"


def _print_help() -> None:
    print("Usage: atomworld [generate|benchmark|eval|draw|visualize] [options]")
    print("\nCommands:")
    print("  generate     Generate AtomWorld benchmark data")
    print("  benchmark    Run benchmark (inference + evaluation by default)")
    print("  eval         Evaluate an existing inference JSON (skip inference)")
    print("  draw         Draw metrics distributions from evaluation_results.json")
    print("  visualize    Alias of draw")
    print("\nTips:")
    print("  atomworld benchmark --help")
    print("  atomworld eval --help")
    print("  atomworld generate --help")
    print("  atomworld draw --help")


def _normalize_eval_args(args: list[str]) -> list[str]:
    normalized = []
    i = 0
    while i < len(args):
        token = args[i]
        if token == "-i":
            normalized.append("--inference_file")
        else:
            normalized.append(token)
        i += 1
    return normalized


def _run_generate(args: list[str]) -> None:
    from data_generation.cif_action_generator import main as generate_main

    generate_main(args)


def _run_benchmark(args: list[str]) -> None:
    from run_benchmark import main as benchmark_main

    benchmark_main(args)


def _run_eval(args: list[str]) -> None:
    from run_benchmark import main as benchmark_main

    normalized = _normalize_eval_args(args)
    if "-h" not in normalized and "--help" not in normalized:
        has_inference_file = any(
            t == "--inference_file" or t.startswith("--inference_file=")
            for t in normalized
        )
        if not has_inference_file:
            print("Error: eval requires --inference_file (or -i).")
            print("Example: atomworld eval -f data -a move_atom_action -i path/to/inference_results.json")
            sys.exit(2)

    benchmark_main(["--skip_inference", *normalized])


def _run_draw(args: list[str]) -> None:
    from utils.args import get_visualization_parser
    from utils.visualization import plot_metrics_distribution

    parser = get_visualization_parser()
    parsed = parser.parse_args(args)

    with open(parsed.input_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    output_folder = parsed.output_folder
    if output_folder is None:
        output_folder = os.path.dirname(os.path.abspath(parsed.input_file))

    plot_metrics_distribution(results, output_folder)
    print(f"Saved plots to: {output_folder}")


def main() -> None:
    print(f"{CYAN}{BANNER_ATOM}{RESET}{MAGENTA}{BANNER_WORLD}{RESET}", end="\n")
    print(f"Version: {_resolve_version()}")
    print("Authors: Taoyuze Lv, Alex, Fengyu Xie")
    print("License: MIT")
    print("-" * 40)

    if len(sys.argv) < 2:
        _print_help()
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command in ["-h", "--help"]:
        _print_help()
        return
    if command == "generate":
        _run_generate(args)
        return
    if command == "benchmark":
        _run_benchmark(args)
        return
    if command == "eval":
        _run_eval(args)
        return
    if command in ["draw", "visualize"]:
        _run_draw(args)
        return

    print(f"Unknown command: {command}")
    _print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()