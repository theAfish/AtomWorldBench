"""
AtomWorld — public Python API.

Lightweight core (only pymatgen + numpy)::

    from atomworld import evaluate, EvaluateResult

Full toolkit (install with ``pip install atomworld[all]``)::

    from atomworld import BenchmarkRunner, CIFActionGenerator, load_data
"""

from atomworld.evaluate import evaluate, EvaluateResult

__all__ = [
    # core — always available
    "evaluate",
    "EvaluateResult",
]


def __getattr__(name: str):
    """Lazy imports for heavy modules that need optional dependencies."""
    if name == "BenchmarkRunner":
        from benchmark.runner import BenchmarkRunner

        return BenchmarkRunner
    if name == "CIFActionGenerator":
        from data_generation.cif_action_generator import CIFActionGenerator

        return CIFActionGenerator
    if name == "load_data":
        from utils.dataloader import load_data

        return load_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
