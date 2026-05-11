"""Modular verifier framework for AtomWorld evaluation.

Verifiers are pluggable, named components that each assess one aspect of a
generated structure against a target.  They are stored by name in dataset
items so that each task family can declare its own verification pipeline.

Usage
-----
    from benchmark.evaluation.verifiers import VerifierRegistry, DEFAULT_VERIFIER_CHAIN

    chain = VerifierRegistry.get_chain(DEFAULT_VERIFIER_CHAIN)
    # or load standard verifiers (import registers them)
    from benchmark.evaluation.verifiers import standard  # noqa: F401
"""

from .base import BaseVerifier, VerificationContext, VerifierResult
from .registry import VerifierRegistry, DEFAULT_VERIFIER_CHAIN
from . import standard  # registers all standard verifiers

__all__ = [
    "BaseVerifier",
    "VerificationContext",
    "VerifierResult",
    "VerifierRegistry",
    "DEFAULT_VERIFIER_CHAIN",
]
