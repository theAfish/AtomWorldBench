"""Implementation of actions that add a motif to a structure."""
from typing import Optional, Tuple
from numbers import Number

import numpy as np
from ase import Atoms
from numpy.typing import ArrayLike

from .base import BaseAction
from ..motifs.base import BaseMotif
from ...utils.description_utils import format_arraylike


class AddMotifAction(BaseAction):
    """Action to add a motif to a structure.

    This action defines how to add a motif to a given structure, including the
    description of the action and the execution logic.
    """
    allowed_relative_styles = [
        "centroid_distance",
        "position_in_line"
    ]
    def __init__(
            self,
            at_position: Optional[ArrayLike] = None,
            relative_to_position: Optional[ArrayLike] = None,
            position_fractional: Optional[bool] = True,
            relative_style: Optional[str] = None,
            relative_to_motif: Optional[BaseMotif] = None,
            relative_shift: Optional[ArrayLike | float] = None,
            relative_atom_index: Optional[int] = 0,
    ):
        """Initialize the AddMotifAction with optional parameters.

        Args:
            at_position (ArrayLike, optional): The position where the motif is added.
                If provided, it overrides all relative parameters.
            relative_to_position (ArrayLike, optional): The position to which the motif
                is added relative to. Only one of `relative_to_position` or
                `relative_to_motif` can be provided.
            position_fractional (bool, optional): Whether all positions are fractional.
                Default is True.
            relative_style (str, optional): The style to determine relative action.
                If not provided, default to `centroid_distance`.
            relative_to_motif (BaseMotif, optional): A motif that the action is taken
                relative to. Only one of `relative_to_position` or `relative_to_motif`
                can be provided.
            relative_shift (ArrayLike or float, optional):
                 A vector or float distance defining the relative position.
            relative_atom_index (int, optional):
                 The index of the atom in the relative motif to insert atom at `relative_shift`
                 distance, if relative_style is `position_in_line`.
                 Default to 0, i.e., the first atom in the pair motif.
        """
        super().__init__(relative_to_motif, relative_style)
        self.position_fractional = position_fractional

        if at_position is not None: # Override all relative parameters.
            at_position = np.array(at_position)
            if at_position.shape != (3,):
                raise ValueError(
                    "at_position must be a 3D vector, got shape: "
                    f"{at_position.shape}"
                )
            self.at_position = at_position
            self.relative_to_position = None
            self.relative_shift = None
            self.relative_atom_index = None
            self.relative_style = None
            self.relative_to_motif = None

        else:
            self.at_position = None
            if relative_shift is None:
                raise ValueError(
                    "relative_shift must be provided when using relative insertion."
                )
            if self.relative_style == "position_in_line":
                if not isinstance(relative_shift, Number):
                    raise ValueError(
                        "relative_shift must be a float when relative_style is 'position_in_line'."
                    )
                self.relative_shift = float(relative_shift)
                if not relative_atom_index in [0, 1]:
                    raise ValueError(
                        "relative_atom_index must be 0 or 1 when relative_style is 'position_in_line'."
                    )
                self.relative_atom_index = relative_atom_index
                if self.relative_to_motif is None:
                    raise ValueError(
                        "relative_to_motif must be provided when relative_style is 'position_in_line'."
                    )
                self.relative_to_motif = relative_to_motif
                self.relative_to_position = None
            else:
                self.relative_shift = np.array(relative_shift)
                self.relative_atom_index = None
                if relative_to_position is not None and relative_to_motif is not None:
                    raise ValueError(
                        "Only one of relative_to_position or relative_to_motif can be provided."
                    )
                if relative_to_position is None and relative_to_motif is None:
                    raise ValueError(
                        "Either relative_to_position or relative_to_motif must be provided."
                    )
                if relative_to_position is not None:
                    relative_to_position = np.array(relative_to_position)
                    if relative_to_position.shape != (3,):
                        raise ValueError(
                            "relative_to_position must be a 3D vector, got shape: "
                            f"{relative_to_position.shape}"
                        )
                self.relative_to_position = relative_to_position
                self.relative_to_motif = relative_to_motif

    def _check_compatibility(self, atoms: Atoms, motif: BaseMotif) -> Tuple[bool, str]:
        """Check if the action is compatible with the given Atoms and motif."""
        if self.relative_style == "position_in_line":
            max_d = self.relative_to_motif.radius * 2  # Bond length.
            if self.relative_shift > max_d:
                return False, (
                    f"relative_shift {self.relative_shift} is larger than "
                    f"the distance {max_d} of atoms in the pair motif."
                )
        return True, ""

    def _compute_insert_cart_position(self, atoms: Atoms):
        """Get inserted cartesian position based on style."""
        # Directly given.
        if self.at_position is not None:
            if self.position_fractional:
                return self.at_position @ atoms.cell
            return self.at_position
        # Compute relative to position.
        elif self.relative_to_position is not None:
            pos = np.array(self.relative_to_position) + self.relative_shift
            if self.position_fractional:
                return pos @ atoms.cell
            return pos
        # Compute relative to motif.
        elif self.relative_to_motif is not None:
            # Insert in line motif.
            if self.relative_style == "position_in_line":
                # Get the position of the atom in the relative motif.
                centroid = self.relative_to_motif.get_positions(wrap=False)[self.relative_atom_index]
                ref_position = self.relative_to_motif.get_positions(wrap=False)[1 - self.relative_atom_index]
                bond_norm_vec = (ref_position - centroid) / np.linalg.norm(ref_position - centroid)
                relative_shift = self.relative_shift * bond_norm_vec
            # Insert at distance to relative motif centroid.
            else:
                centroid = self.relative_to_motif.get_centroid(
                    fractional=False
                )
                if self.position_fractional:
                    relative_shift = self.relative_shift @ atoms.cell
                else:
                    relative_shift = self.relative_shift
            return centroid + relative_shift
        else:
            raise ValueError("No valid position provided for motif addition.")


    def _execute(self, atoms: Atoms, operated_motif: BaseMotif) -> Atoms:
        """Execute the action on the structure to generate the ground truth structure."""
        insert_position = self._compute_insert_cart_position(atoms)
        displacement = insert_position - operated_motif.get_centroid(fractional=False)
        operated_motif.translate(displacement)
        new_atoms = atoms.copy()
        new_atoms += operated_motif.get_atoms()
        return new_atoms

    def describe(
            self,
            motif: BaseMotif,
            precision: int = 4,
            motif_kwargs: Optional[dict] = None,
            relative_motif_kwargs: Optional[dict] = None,
    ) -> str:
        """Describe the action for LLM prompting.

        Args:
            motif (BaseMotif): The motif being added.
            precision (int): The precision for formatting numerical values in the description.
                Will overwrite motif and relative motif description precision settings.
            motif_kwargs (dict, optional): Additional keyword arguments for the motif description.
            relative_motif_kwargs (dict, optional): Additional keyword arguments for the relative motif description.
        Returns:
            str: A description of the action.
        """
        motif_kwargs = motif_kwargs or {}
        relative_motif_kwargs = relative_motif_kwargs or {}

        motif_kwargs.update({"precision": precision})
        relative_motif_kwargs.update({"precision": precision})

        if self.position_fractional:
            coord_word = "fractional coordinates"
        else:
            coord_word = "cartesian coordinates"
        if self.at_position is not None:
            return (f"add [{motif.describe(**motif_kwargs)}], with its center located at {coord_word}"
                    f" {format_arraylike(self.at_position, precision=precision)}")
        elif self.relative_to_position is not None:
            return (
                f"add [{motif.describe(**motif_kwargs)}], with its center shifted in {coord_word} by"
                f" [{format_arraylike(self.relative_shift, precision=precision)}] relative to a"
                f" reference point at {coord_word}"
                f" {format_arraylike(self.relative_to_position, precision=precision)}"
            )
        elif self.relative_to_motif is not None:
            if self.relative_style == "position_in_line":
                relative_motif_kwargs.update({"style": "index"})
                return (
                    f"add [{motif.describe(**motif_kwargs)}], with its center located on the line between"
                    f" by [{self.relative_to_motif.describe(**relative_motif_kwargs)}], at"
                    f" {self.relative_shift:.{precision}f} angstroms away from the atom indexed"
                    f" {self.relative_to_motif.indices[self.relative_atom_index]}"
                )
            else:
                return (
                    f"add [{motif.describe(**motif_kwargs)}], with its center shifted in {coord_word} by"
                    f" {format_arraylike(self.relative_shift, precision=precision)} relative to the"
                    f" centroid of [{self.relative_to_motif.describe(**relative_motif_kwargs)}]"
                )
        else:
            raise ValueError("No valid position provided for motif addition.")