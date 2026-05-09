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
    print("Usage: atomworld [generate|benchmark|eval|draw|visualize|serve] [options]")
    print("\nCommands:")
    print("  generate     Generate AtomWorld benchmark data")
    print("  benchmark    Run benchmark (inference + evaluation by default)")
    print("  eval         Evaluate an existing inference JSON (skip inference)")
    print("  draw         Draw metrics distributions from evaluation_results.json")
    print("  visualize    Alias of draw")
    print("  serve        Start the AtomWorldBench REST API server")
    print("\nTips:")
    print("  atomworld benchmark --help")
    print("  atomworld eval --help")
    print("  atomworld generate --help")
    print("  atomworld draw --help")
    print("  atomworld serve --help")


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


def _run_serve(args: list[str]) -> None:
    import argparse
    import os
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the AtomWorldBench REST API server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=50001, help="Port to listen on (default: 50001)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Bootstrap admin API key used for privileged requests and fallback authentication",
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        default="data",
        help="Path to the benchmark data folder (default: data)",
    )
    parser.add_argument(
        "--sessions-dir",
        type=str,
        default="sessions",
        help="Directory to persist session state and results (default: sessions)",
    )
    parsed = parser.parse_args(args)

    os.environ["ATOMWORLD_API_KEY"] = parsed.api_key

    from api.server import create_app

    app = create_app(
        data_folder=parsed.data_folder,
        sessions_dir=parsed.sessions_dir,
    )
    print(f"Starting AtomWorldBench API on http://{parsed.host}:{parsed.port}")
    print(f"Public entry : http://<server-host>:{parsed.port}/")
    print(f"Access info  : http://<server-host>:{parsed.port}/access-info")
    print(f"OpenAPI docs : http://<server-host>:{parsed.port}/docs")
    print(f"Data folder : {parsed.data_folder}")
    print(f"Sessions dir: {parsed.sessions_dir}")
    uvicorn.run(app, host=parsed.host, port=parsed.port)


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
    if command == "serve":
        _run_serve(args)
        return

    print(f"Unknown command: {command}")
    _print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()