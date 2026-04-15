"""
Standalone evaluate function for AtomWorld.

Only depends on pymatgen + numpy, so it stays lightweight for use as an
RL reward / judge function without pulling in torch, transformers, etc.

Usage
-----
    from atomworld import evaluate

    result = evaluate(target_cif=ground_truth_cif, generated_output=model_output)
    print(result.correct, result.rmsd)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifParser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvaluateResult:
    """Result of evaluating a generated CIF against a target."""

    correct: bool
    wrong_type: Optional[str] = None
    rmsd: Optional[float] = None
    max_dist: Optional[float] = None


# ---------------------------------------------------------------------------
# Internal helpers (duplicated from utils/ so the import stays light)
# ---------------------------------------------------------------------------

def _extract_cif(text: str) -> Optional[str]:
    """Extract CIF content from the last ``<cif>…</cif>`` block in *text*."""
    start_tag = "<cif>"
    end_tag = "</cif>"
    start = text.rfind(start_tag)
    end = text.rfind(end_tag, start)
    if start == -1 or end == -1:
        return None
    return text[start + len(start_tag):end].strip()


def _parse_cif(cif_string: str, primitive: bool = False):
    """Parse a CIF string into a :class:`pymatgen.core.Structure`."""
    try:
        parser = CifParser.from_str(cif_string)
        structures = parser.parse_structures(primitive=primitive, check_occu=False)
        return structures[0] if structures else None
    except Exception as e:
        logger.debug("CIF parsing error: %s", e)
        return None


def _check_atom_counts(struct1, struct2) -> bool:
    """Return *True* if both structures have the same element counts."""
    if len(struct1) != len(struct2):
        return False
    return struct1.composition.as_dict() == struct2.composition.as_dict()


def _match_structures(struct1, struct2, primitive_cell=False, stol=0.5):
    """Match via :class:`StructureMatcher`. Returns ``(rmsd, max_dist)``."""
    matcher = StructureMatcher(primitive_cell=primitive_cell, stol=stol)
    if matcher.fit(struct1, struct2):
        return matcher.get_rms_dist(struct1, struct2)
    return None, None


def _exact_match_metrics(struct1, struct2, stol=0.5):
    """RMSD / max-dist with minimum-image PBC for identical-lattice structures."""
    if len(struct1) != len(struct2):
        return None, None

    for s1, s2 in zip(struct1.sites, struct2.sites):
        if s1.species_string != s2.species_string:
            return -1, -1

    lat1, lat2 = struct1.lattice, struct2.lattice
    if not np.allclose(lat1.matrix, lat2.matrix):
        return -1, -1

    frac1 = np.array([s.frac_coords for s in struct1.sites])
    frac2 = np.array([s.frac_coords for s in struct2.sites])

    diff = frac1 - frac2
    diff -= np.round(diff)

    cart_diff = diff @ lat1.matrix
    sq = np.sum(cart_diff**2, axis=1)
    norm = (len(struct1) / lat1.volume) ** (1 / 3)
    rmsd = float(np.sqrt(np.mean(sq)) * norm)
    max_dist = float(np.sqrt(np.max(sq)) * norm)

    if max_dist > stol:
        return -1, -1

    return rmsd, max_dist


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    target_cif: str,
    generated_output: str,
    *,
    action_name: str | None = None,
) -> EvaluateResult:
    """Evaluate a model-generated CIF against a ground-truth target.

    Parameters
    ----------
    target_cif : str
        Ground-truth CIF content (plain text, **no** ``<cif>`` tags needed).
    generated_output : str
        Raw model output.  The CIF is extracted from ``<cif>…</cif>`` tags
        automatically.  If you are passing a bare CIF string, wrap it first::

            generated_output = f"<cif>{my_cif}</cif>"

    action_name : str, optional
        Action name such as ``"move_all_action"``.  Selects the metric:

        * ``"move_all_action"`` → exact-match positional RMSD (PBC-aware).
        * anything else (default) → ``StructureMatcher``-based RMSD.

    Returns
    -------
    EvaluateResult
        ``correct``    – whether the generated structure matches the target.
        ``wrong_type`` – error category string if incorrect, else ``None``.
        ``rmsd``       – root-mean-square displacement (Å, if correct).
        ``max_dist``   – maximum single-atom displacement (Å, if correct).
    """
    # ---- target ----
    target_struct = _parse_cif(target_cif, primitive=False)
    if target_struct is None:
        raise ValueError("Could not parse the target CIF string.")

    # ---- generated ----
    gen_cif = _extract_cif(generated_output)
    if gen_cif is None:
        return EvaluateResult(correct=False, wrong_type="OutputFormatError")

    gen_struct = _parse_cif(gen_cif, primitive=False)
    if gen_struct is None:
        return EvaluateResult(correct=False, wrong_type="CIFParsingError")

    # ---- atom-count check ----
    if not _check_atom_counts(target_struct, gen_struct):
        return EvaluateResult(correct=False, wrong_type="AtomCountMismatch")

    # ---- structural matching ----
    use_exact = (action_name or "").lower() == "move_all_action"
    if use_exact:
        rmsd, max_dist = _exact_match_metrics(target_struct, gen_struct)
    else:
        rmsd, max_dist = _match_structures(
            target_struct, gen_struct, primitive_cell=False
        )

    if rmsd is None or rmsd == -1:
        return EvaluateResult(correct=False, wrong_type="StructureMismatch")

    return EvaluateResult(correct=True, rmsd=rmsd, max_dist=max_dist)
