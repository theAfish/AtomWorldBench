"""Base classes for the verifier framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VerificationContext:
    """Mutable context shared across verifiers in a chain.

    Verifiers may read *and write* this object so that earlier steps can
    populate derived values (e.g. extracted CIF text, parsed structure) for
    later steps to consume without repeating work.
    """

    # Raw model output (text) — None in pure agent-file mode
    generated_output: Optional[str]
    # Extracted CIF text — set by OutputFormatVerifier (or pre-set by evaluator
    # in agent mode from the file on disk)
    generated_cif: Optional[str]
    # Parsed pymatgen Structure — set by CIFParsingVerifier
    generated_structure: Optional[Any]
    # Ground-truth pymatgen Structure (always present)
    target_structure: Any
    # Task-specific metadata stored in the dataset (e.g. molecule indices)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Full inference-result item (for any extra lookups)
    item: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierResult:
    """Result returned by a single verifier."""

    name: str
    passed: bool
    # Normalised score in [0, 1]; typically 0 or 1 for binary verifiers
    score: float
    # Short error-type string if not passed (mirrors AtomWorldEvaluator convention)
    wrong_type: Optional[str] = None
    # Optional structured details (e.g. {"rmsd": 0.02, "max_dist": 0.05})
    details: Optional[Dict[str, Any]] = None


class BaseVerifier(ABC):
    """Abstract base class for all verifiers.

    Subclasses must set ``name`` and implement ``verify``.

    Attributes
    ----------
    name : str
        Registry key.  Must be unique across all registered verifiers.
    blocking : bool
        When *True* (default), a failure stops the remainder of the chain.
        Set to *False* for informational verifiers that should always run.
    """

    name: str = ""
    blocking: bool = True

    @abstractmethod
    def verify(self, ctx: VerificationContext) -> VerifierResult:
        """Assess one aspect of the generated structure.

        Parameters
        ----------
        ctx:
            Shared mutable context.  May be modified (e.g. to set
            ``generated_cif`` or ``generated_structure``) so subsequent
            verifiers can reuse the results.

        Returns
        -------
        VerifierResult
        """
