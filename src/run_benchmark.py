#!/usr/bin/env python3
"""
Unified benchmark runner for all benchmark types.
Supports AtomWorld, PointWorld, CIFGen, and CIFRepair benchmarks.
"""

from benchmark.parser import BenchmarkArgumentParser
from benchmark.config import BenchmarkConfig
from benchmark.runner import BenchmarkRunner

def main():
    # Parse and validate arguments
    args = BenchmarkArgumentParser.parse_and_validate()
    
    # Create benchmark configuration
    config = BenchmarkConfig.from_args(args)
    
    # Create and run the benchmark
    runner = BenchmarkRunner(config)
    runner.run()

if __name__ == "__main__":
    main()