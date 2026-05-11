"""RemoveMoleculeAction — active task for surface adsorbate removal.

Task definition
---------------
Given a periodic slab structure with one or more adsorbed molecules, the
model must produce the clean slab (all adsorbate atoms removed).

Prompt (invariant across instances)
-------------------------------------
"Remove the molecule(s) inside the structure."

Evaluation verifier chain
--------------------------
output_format → cif_parsing → atom_count → structure_match

The default chain is the same as for simple tasks: the model must output a
valid CIF whose composition and geometry match the clean slab target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Verifier chain for this task — stored verbatim in each dataset item.
REMOVE_MOLECULE_VERIFIERS: List[str] = [
    "output_format",
    "cif_parsing",
    "atom_count",
    "structure_match",
]

ACTION_PROMPT = "Remove the molecule(s) inside the structure."


@dataclass
class RemoveMoleculeAction:
    """Represents a single RemoveMolecule task instance.

    Attributes
    ----------
    molecule_formula : str
        Chemical formula of the adsorbed molecule (e.g. ``"H2O"``).
    molecule_num_atoms : int
        Number of atoms in the molecule.
    molecule_indices : list[int]
        0-based indices of the adsorbate atoms in the *input* (slab +
        adsorbate) structure.
    slab_composition : dict[str, int]
        Elemental composition of the clean slab target (for reference).
    miller_index : list[int]
        Miller index used when cutting the slab from the bulk structure.
    """

    molecule_formula: str
    molecule_num_atoms: int
    molecule_indices: List[int]
    slab_composition: Dict[str, int] = field(default_factory=dict)
    miller_index: List[int] = field(default_factory=lambda: [0, 0, 1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Return the natural-language action prompt."""
        return ACTION_PROMPT

    def __str__(self) -> str:
        return self.describe()

    def get_metadata(self) -> Dict[str, Any]:
        """Return serialisable metadata for the dataset item."""
        return {
            "molecule_formula": self.molecule_formula,
            "molecule_num_atoms": self.molecule_num_atoms,
            "molecule_indices": self.molecule_indices,
            "slab_composition": self.slab_composition,
            "miller_index": self.miller_index,
        }

    @staticmethod
    def get_verifiers() -> List[str]:
        """Return the verifier chain for this action family."""
        return list(REMOVE_MOLECULE_VERIFIERS)
