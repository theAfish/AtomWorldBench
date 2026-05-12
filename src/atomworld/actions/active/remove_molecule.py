"""RemoveMoleculeAction — active task for molecule removal.

Two task variants are generated and randomly mixed at dataset-creation time:

* **surface** — a molecule is adsorbed on a periodic slab; the model must
  produce the clean slab (all adsorbate atoms removed).
* **bulk** — a molecule is inserted into a bulk supercell with nearby
  host atoms deleted to make room; the model must produce the cavitated
  bulk structure (molecule removed, vacancies kept).

The natural-language prompt is identical for both variants:
  "Remove the molecule(s) inside the structure."

Evaluation verifier chain (both variants)
------------------------------------------
output_format → cif_parsing → atom_count → structure_match
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
        Chemical formula of the molecule (e.g. ``"H2O"``).
    molecule_num_atoms : int
        Number of atoms in the molecule.
    molecule_indices : list[int]
        0-based indices of the molecule atoms in the *input* structure.
    task_variant : str
        ``"surface"`` — molecule adsorbed on a periodic slab.
        ``"bulk"``    — molecule inserted into a 3-D periodic bulk supercell.
    slab_composition : dict[str, int]
        Elemental composition of the clean-slab target (surface variant only).
    miller_index : list[int]
        Miller index used when cutting the slab (surface variant only).
    bulk_composition : dict[str, int]
        Elemental composition of the cavitated bulk (bulk variant only).
    n_removed_atoms : int
        Number of host atoms deleted to make room for the molecule
        (bulk variant only).
    supercell_repeat : int
        Isotropic repeat factor used for the bulk supercell (bulk variant only).
    """

    molecule_formula: str
    molecule_num_atoms: int
    molecule_indices: List[int]
    task_variant: str = "surface"
    slab_composition: Dict[str, int] = field(default_factory=dict)
    miller_index: List[int] = field(default_factory=lambda: [0, 0, 1])
    bulk_composition: Dict[str, int] = field(default_factory=dict)
    n_removed_atoms: int = 0
    supercell_repeat: int = 1

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
        base = {
            "task_variant": self.task_variant,
            "molecule_formula": self.molecule_formula,
            "molecule_num_atoms": self.molecule_num_atoms,
            "molecule_indices": self.molecule_indices,
        }
        if self.task_variant == "surface":
            base.update({
                "slab_composition": self.slab_composition,
                "miller_index": self.miller_index,
            })
        else:
            base.update({
                "bulk_composition": self.bulk_composition,
                "n_removed_atoms": self.n_removed_atoms,
                "supercell_repeat": self.supercell_repeat,
            })
        return base

    @staticmethod
    def get_verifiers() -> List[str]:
        """Return the verifier chain for this action family."""
        return list(REMOVE_MOLECULE_VERIFIERS)
