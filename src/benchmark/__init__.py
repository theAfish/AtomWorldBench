from .config import BenchmarkConfig
from .parser import BenchmarkArgumentParser
from .inference import BaseInferencer, AtomWorldInferencer
from .evaluation import BaseOfflineEvaluator, AtomWorldEvaluator

__all__ = [
    'BenchmarkConfig',
    'BenchmarkArgumentParser',
    'BenchmarkRunner',
    'BaseInferencer',
    'AtomWorldInferencer',
    'BaseOfflineEvaluator',
    'AtomWorldEvaluator',
]


def __getattr__(name):
    """Lazy import for BenchmarkRunner to avoid heavy dependency chain."""
    if name == 'BenchmarkRunner':
        from .runner import BenchmarkRunner
        return BenchmarkRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")