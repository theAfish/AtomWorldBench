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
        try:
            from benchmark.runner import BenchmarkRunner
        except ImportError as exc:
            raise AttributeError(
                "BenchmarkRunner requires optional benchmark dependencies. "
                "Install them with `pip install atomworld[benchmark]`."
            ) from exc

        return BenchmarkRunner
    if name == "CIFActionGenerator":
        try:
            from data_generation.cif_action_generator import CIFActionGenerator
        except ImportError as exc:
            raise AttributeError(
                "CIFActionGenerator requires optional data-generation dependencies. "
                "Install them with `pip install atomworld[all]`."
            ) from exc

        return CIFActionGenerator
    if name == "load_data":
        try:
            from utils.dataloader import load_data
        except ImportError as exc:
            raise AttributeError(
                "load_data requires optional data-loading dependencies. "
                "Install them with `pip install atomworld[all]`."
            ) from exc

        return load_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
