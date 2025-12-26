import sys
import importlib.metadata
from AtomWorldBench.benchmark.run_benchmark import main as benchmark_main
from AtomWorldBench.data_generation.cif_action_generator import main as generate_main
from AtomWorldBench.scripts.visualize_results import main as visualize_main

BANNER = r"""
   ___  __                 
  / _ |/ /____  __ _       
 / __ / __/ _ \/  ' \      
/_/ |_\__/\___/_/_/_/_   __
 | | /| / /__  ____/ /__/ /
 | |/ |/ / _ \/ __/ / _  / 
 |__/|__/\___/_/ /_/\_,_/  
                           """

def main():
    print(BANNER)
    
    try:
        version = importlib.metadata.version("atom_world")
    except importlib.metadata.PackageNotFoundError:
        try:
            version = importlib.metadata.version("AtomWorldBench")
        except importlib.metadata.PackageNotFoundError:
            version = "0.1.0 (dev)"

    print(f"Version: {version}")
    print("Authors: Taoyuze Lv, Alex, Fengyu Xie")
    print("License: MIT")
    print("-" * 40)

    if len(sys.argv) < 2:
        print("Usage: atomworld [benchmark|generate|visualize] [options]")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "benchmark":
        benchmark_main(args)
    elif command == "generate":
        generate_main(args)
    elif command == "visualize":
        visualize_main(args)
    elif command in ["-h", "--help"]:
        print("Usage: atomworld [benchmark|generate|visualize] [options]")
        print("\nCommands:")
        print("  benchmark    Run AtomWorld Benchmark")
        print("  generate     Generate AtomWorld Benchmark Data")
        print("  visualize    Visualize AtomWorld Benchmark Results")
    else:
        print(f"Unknown command: {command}")
        print("Usage: atomworld [benchmark|generate|visualize] [options]")
        sys.exit(1)

if __name__ == "__main__":
    main()
