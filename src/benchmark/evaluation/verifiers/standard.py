"""Standard verifiers that replicate the existing simple/verbose task logic.

Importing this module automatically registers all verifiers in the global
``VerifierRegistry``.  The ``benchmark.evaluation.verifiers`` package imports
this module, so it is sufficient to import the package.

Registered names (in default chain order)
------------------------------------------
output_format
    Extracts the CIF block from tagged model output.  Also accepts raw CIF
    text (agent mode).  Sets ``ctx.generated_cif`` on success.
cif_parsing
    Parses ``ctx.generated_cif`` into a pymatgen Structure.
    Sets ``ctx.generated_structure`` on success.
atom_count
    Checks that generated and target structures share the same elemental
    composition.
structure_match
    Checks structural similarity via ``StructureMatcher``.
exact_structure_match
    Stricter positional-RMSD check used for move_all_action tasks.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import BaseVerifier, VerificationContext, VerifierResult
from .registry import VerifierRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — these are only available in the benchmark extras
# ---------------------------------------------------------------------------

def _import_dataloader():
    from utils.dataloader import load_cif_file_from_string
    return load_cif_file_from_string


def _import_extract():
    from utils.extract_data import extract_from_string
    return extract_from_string


def _import_metrics():
    from benchmark.evaluation.metrics import (
        check_atom_counts,
        match_structures,
        compute_exact_match_positional_metrics,
    )
    return check_atom_counts, match_structures, compute_exact_match_positional_metrics


# ---------------------------------------------------------------------------
# output_format
# ---------------------------------------------------------------------------


@VerifierRegistry.register("output_format")
class OutputFormatVerifier(BaseVerifier):
    """Check that the model output contains a readable CIF block.

    Pipeline effects
    ----------------
    Sets ``ctx.generated_cif`` when a CIF block is found so that
    ``CIFParsingVerifier`` (and any later verifier) can reuse it.

    If ``ctx.generated_cif`` is already populated (e.g. pre-set by the
    evaluator in agent-file mode), the verifier passes immediately without
    touching ``ctx.generated_output``.
    """

    name = "output_format"
    blocking = True

    def verify(self, ctx: VerificationContext) -> VerifierResult:
        # Agent-file mode: CIF was already read from disk by the evaluator.
        if ctx.generated_cif is not None:
            return VerifierResult(name=self.name, passed=True, score=1.0)

        generated_output = ctx.generated_output
        if not isinstance(generated_output, str):
            return VerifierResult(
                name=self.name,
                passed=False,
                score=0.0,
                wrong_type="OutputMissing",
            )

        extract_from_string = _import_extract()

        # Try <cif>…</cif> tags first (LLM mode).
        extracted = extract_from_string(generated_output, format="cif")
        if extracted is not None:
            ctx.generated_cif = extracted
            return VerifierResult(name=self.name, passed=True, score=1.0)

        # Fallback: bare CIF text without tags (some agent outputs).
        raw = generated_output.strip()
        if raw and "data_" in raw and "_atom_site" in raw:
            ctx.generated_cif = raw
            return VerifierResult(name=self.name, passed=True, score=1.0)

        return VerifierResult(
            name=self.name,
            passed=False,
            score=0.0,
            wrong_type="OutputFormatError",
        )


# ---------------------------------------------------------------------------
# cif_parsing
# ---------------------------------------------------------------------------


@VerifierRegistry.register("cif_parsing")
class CIFParsingVerifier(BaseVerifier):
    """Parse ``ctx.generated_cif`` into a pymatgen Structure.

    Pipeline effects
    ----------------
    Sets ``ctx.generated_structure`` on success.
    """

    name = "cif_parsing"
    blocking = True

    def verify(self, ctx: VerificationContext) -> VerifierResult:
        if ctx.generated_cif is None:
            return VerifierResult(
                name=self.name,
                passed=False,
                score=0.0,
                wrong_type="CIFMissing",
            )

        load_cif_file_from_string = _import_dataloader()

        try:
            struct = load_cif_file_from_string(ctx.generated_cif, primitive=False)
        except Exception as exc:
            logger.debug("CIF parsing error: %s", exc)
            struct = None

        if struct is None:
            return VerifierResult(
                name=self.name,
                passed=False,
                score=0.0,
                wrong_type="CIFParsingError",
            )

        ctx.generated_structure = struct
        return VerifierResult(name=self.name, passed=True, score=1.0)


# ---------------------------------------------------------------------------
# atom_count
# ---------------------------------------------------------------------------


@VerifierRegistry.register("atom_count")
class AtomCountVerifier(BaseVerifier):
    """Check that generated and target structures have the same composition."""

    name = "atom_count"
    blocking = True

    def verify(self, ctx: VerificationContext) -> VerifierResult:
        if ctx.generated_structure is None or ctx.target_structure is None:
            return VerifierResult(
                name=self.name,
                passed=False,
                score=0.0,
                wrong_type="StructureMissing",
            )

        check_atom_counts, _, _ = _import_metrics()

        if not check_atom_counts(ctx.target_structure, ctx.generated_structure):
            return VerifierResult(
                name=self.name,
                passed=False,
                score=0.0,
                wrong_type="AtomCountMismatch",
            )

        return VerifierResult(name=self.name, passed=True, score=1.0)


# ---------------------------------------------------------------------------
# structure_match
# ---------------------------------------------------------------------------


@VerifierRegistry.register("structure_match")
class StructureMatchVerifier(BaseVerifier):
    """Structural similarity check via ``StructureMatcher``."""

    name = "structure_match"
    blocking = True

    def verify(self, ctx: VerificationContext) -> VerifierResult:
        if ctx.generated_structure is None or ctx.target_structure is None:
            return VerifierResult(
                name=self.name,
                passed=False,
                score=0.0,
                wrong_type="StructureMissing",
            )

        _, match_structures, _ = _import_metrics()

        rmsd, max_dist = match_structures(
            ctx.target_structure,
            ctx.generated_structure,
            primitive_cell=False,
        )

        if rmsd is None or rmsd == -1:
            return VerifierResult(
                name=self.name,
                passed=False,
                score=0.0,
                wrong_type="StructureMismatch",
                details={"rmsd": rmsd, "max_dist": max_dist},
            )

        return VerifierResult(
            name=self.name,
            passed=True,
            score=1.0,
            details={"rmsd": float(rmsd), "max_dist": float(max_dist)},
        )


# ---------------------------------------------------------------------------
# exact_structure_match  (used by move_all_action)
# ---------------------------------------------------------------------------


@VerifierRegistry.register("exact_structure_match")
class ExactStructureMatchVerifier(BaseVerifier):
    """Positional-RMSD check with PBC, assuming identical lattices."""

    name = "exact_structure_match"
    blocking = True

    def verify(self, ctx: VerificationContext) -> VerifierResult:
        if ctx.generated_structure is None or ctx.target_structure is None:
            return VerifierResult(
                name=self.name,
                passed=False,
                score=0.0,
                wrong_type="StructureMissing",
            )

        _, _, compute_exact = _import_metrics()

        rmsd, max_dist = compute_exact(
            ctx.target_structure, ctx.generated_structure
        )

        if rmsd is None or rmsd == -1:
            return VerifierResult(
                name=self.name,
                passed=False,
                score=0.0,
                wrong_type="StructureMismatch",
                details={"rmsd": rmsd, "max_dist": max_dist},
            )

        return VerifierResult(
            name=self.name,
            passed=True,
            score=1.0,
            details={"rmsd": float(rmsd), "max_dist": float(max_dist)},
        )
