"""Implementation of actions that add a motif to a structure."""
from typing import Optional, Tuple
from numbers import Number

import numpy as np
from ase import Atoms
from numpy.typing import ArrayLike

from .base import BaseAction
from ..motifs.site_collections.base import BaseSiteCollectionMotif
from ...utils.description_utils import describe_arraylike
from ...utils.coord_utils import check_coordinates_shape

from ...globals import DEFAULT_FLOAT_TO_STRING_PRECISION


def _check_relative_shift(x):
    if isinstance(x, Number):
        return float(x)
    else:
        return check_coordinates_shape(
            x, "relative_shift", expected_1d=True, allow_none=True
        )


class AddMotifAction(BaseAction):
    """Action to add a motif to a structure.

    This action defines how to add a motif to a given structure, including the
    description of the action and the execution logic.
    """
    kwargs_and_formating_functions = {
        "at_position":
            lambda x: check_coordinates_shape(
                x, "at_position", expected_1d=True, allow_none=True
            ),
        "relative_to_position":
            lambda x: check_coordinates_shape(
                x, "relative_to_position", expected_1d=True, allow_none=True
            ),
        "relative_shift": _check_relative_shift,
    }
    mode_definitions = {
        "_excluded": ["position_fractional"],
        "absolute": {"at_position": None},
        "relative_to_position": {
            "relative_to_position": None, "relative_shift": None
        },
        "relative_to_regular_motif": {
            "relative_to_motif": None, "relative_shift": None,
            "relative_style": (
                lambda s: s  == "centroid_distance",
                "Relative style must be centroid_distance for relative_to_regular_motif mode."
            ),
        },
        "relative_to_pair_motif": {
            "relative_to_motif": (
                lambda m: len(m) == 2,
                "Only pair motifs are allowed for relative_to_pair_motif mode."
            ),
            "relative_shift": (
                lambda n: isinstance(n, Number),
                "Relative shift must be a number for relative_to_pair_motif mode."
            ),
            "relative_style": (
                lambda s: s == "position_in_line",
                "Relative style must be position_in_line for relative_to_pair_motif mode."
            ),
            "relative_atom_index": (
                lambda i: i in [0, 1],
                "Relative atom index must be provided as 0 or 1 for"
                " relative_to_pair_motif mode."
            ),
        }
    }
    def __init__(
            self,
            at_position: Optional[ArrayLike] = None,
            relative_to_position: Optional[ArrayLike] = None,
            position_fractional: bool = True,
            relative_style: Optional[str] = None,
            relative_to_motif: Optional[BaseSiteCollectionMotif] = None,
            relative_shift: Optional[ArrayLike | float] = None,
            relative_atom_index: Optional[int] = None,
    ):
        """Initialize the AddMotifAction with optional parameters.

        Currently, allows 4 modes of operation:
            1, `absolute`: Add the motif at a specified position in the structure.
                In this mode, only `at_position` is provided, no other parameters
                should be given except position_fractional.
            2, `relative_to_position`: Add the motif relative to a specified position.
                In this mode, provide `relative_to_position` and `relative_shift`.
                No other parameters should be given except position_fractional.
            3, `relative_to_regular_motif`: Add the motif relative to a specified motif's
                centroid. In this mode, provide `relative_to_motif`, `relative_shift`,
                and `relative_style`=="centroid_distance".
                No other parameters should be given except
                position_fractional. Relative motif can be any motif that supports centroid
                relative style.
            4, `relative_to_pair_motif`: Add the motif on a pair motif's line. In this
                mode, provide `relative_to_motif`, `relative_shift`,
                `relative_style` == "position_in_line", and `relative_atom_index`.
                No other parameters should be given except position_fractional.
        Args:
            at_position (ArrayLike, optional): The position where the motif is added.
                If provided, it overrides all relative parameters.
            relative_to_position (ArrayLike, optional): The position to which the motif
                is added relative to.
            position_fractional (bool, optional):
                Whether all positions provided in arguments are fractional. If False,
                will be cartesian.
                This will also affect the description style of the action. Default is True.
            relative_style (str, optional): The style to determine relative action.
            relative_to_motif (BaseMotif, optional): A motif that the action is taken
                relative to.
            relative_shift (ArrayLike or float, optional):
                 A vector or float distance defining the relative position.
            relative_atom_index (int, optional):
                 The index of the atom in the relative motif to insert atom at `relative_shift`
                 distance, if relative_style is `position_in_line`. Must be provided if working
                 in `position_in_line` mode.
        """
        # Static declaration for IDE linting.
        self.at_position = None
        self.relative_to_position = None
        self.position_fractional = None
        self.relative_style = None
        self.relative_to_motif = None
        self.relative_shift = None
        self.relative_atom_index = None

        super().__init__(
            at_position = at_position,
            relative_to_position = relative_to_position,
            position_fractional = position_fractional,
            relative_style = relative_style,
            relative_to_motif = relative_to_motif,
            relative_shift = relative_shift,
            relative_atom_index = relative_atom_index,
        )

    def _check_compatibility(self, atoms: Atoms, motif: BaseSiteCollectionMotif) -> Tuple[bool, str]:
        """Check if the action is compatible with the given Atoms and motif."""
        if self.relative_style == "position_in_line":
            max_d = self.relative_to_motif.radius * 2  # Bond length.
            if self.relative_shift > max_d:
                return False, (
                    f"relative_shift {self.relative_shift} is larger than "
                    f"the distance {max_d} of atoms in the reference pair motif."
                )
        return True, ""

    def _compute_insert_cart_position(self, atoms: Atoms):
        """Get inserted cartesian position based on style."""
        # Directly given.
        if self.mode_flag == "absolute":
            if self.position_fractional:
                return self.at_position @ atoms.cell.complete()
            return self.at_position
        # Compute relative to position.
        elif self.mode_flag == "relative_to_position":
            pos = np.array(self.relative_to_position) + self.relative_shift
            if self.position_fractional:
                return pos @ atoms.cell.complete()
            return pos
        # Compute relative to motif.
        elif self.mode_flag == "relative_to_pair_motif":
                # Get the position of the atom in the relative motif.
                centroid = self.relative_to_motif.get_positions(wrap=False)[self.relative_atom_index]
                ref_position = self.relative_to_motif.get_positions(wrap=False)[1 - self.relative_atom_index]
                bond_norm_vec = (ref_position - centroid) / np.linalg.norm(ref_position - centroid)
                relative_shift = self.relative_shift * bond_norm_vec
        # Insert at distance to relative motif centroid.
        elif self.mode_flag == "relative_to_regular_motif":
            centroid = self.relative_to_motif.get_centroid(
                fractional=False
            )
            if self.position_fractional:
                relative_shift = self.relative_shift @ atoms.cell.complete()
            else:
                relative_shift = self.relative_shift
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}")
        return centroid + relative_shift

    def _execute(self, atoms: Atoms, operated_motif: BaseSiteCollectionMotif) -> Atoms:
        """Execute the action on the structure to generate the ground truth structure.

        Added motif will always be appended to the end of the structure.
        Args:
            atoms (Atoms): The structure to operate on.
            operated_motif (BaseSiteCollectionMotif): The motif to be added to the structure.
        Returns:
            Atoms: The modified structure with the motif added.
        """
        insert_position = self._compute_insert_cart_position(atoms)
        displacement = insert_position - operated_motif.get_centroid(fractional=False)
        operated_motif.translate(displacement)
        new_atoms = atoms.copy()
        new_atoms += operated_motif.get_atoms()
        return new_atoms

    def describe(
            self,
            motif: BaseSiteCollectionMotif,
            precision: int = DEFAULT_FLOAT_TO_STRING_PRECISION,
            motif_kwargs: Optional[dict] = None,
            relative_motif_kwargs: Optional[dict] = None,
            **kwargs  # Just for linting.
    ) -> str:
        """Describe the action for LLM prompting.

        Args:
            motif (BaseSiteCollectionMotif): The motif being added.
            precision (int): The precision for formatting numerical values in the description in decimals.
                Will overwrite motif and relative motif description precision settings.
                Default is set in `globals.py`, typically 4.
            motif_kwargs (dict, optional): Additional keyword arguments for the motif description.
            relative_motif_kwargs (dict, optional): Additional keyword arguments for the relative motif description.
                 Note that motif and relative motif description styles are not affected by the action's
                `position_fractional` attribute.
        Returns:
            str: A description of the action.
        """
        motif_kwargs = motif_kwargs or {}
        relative_motif_kwargs = relative_motif_kwargs or {}

        motif_kwargs.update({"precision": precision, "is_addition": True})
        relative_motif_kwargs.update({"precision": precision, "is_addition": False})

        if self.position_fractional:
            coord_word = "fractional coordinates"
        else:
            coord_word = "cartesian coordinates"

        # Common instruction to guarantee order of addition.
        common_instruction = "newly added motif should be appended to the end of the structure."

        if self.mode_flag == "absolute":
            return (
                    f"add [{motif.describe(**motif_kwargs)}] to the structure,"
                    f" with its centroid located at {coord_word}"
                    f" {describe_arraylike(self.at_position, precision=precision)}."
                    + " " + common_instruction
            )
        if self.mode_flag == "relative_to_position":
            return (
                f"add [{motif.describe(**motif_kwargs)}] to the structure,"
                f" with its centroid shifted in {coord_word} by"
                f" {describe_arraylike(self.relative_shift, precision=precision)} relative to a"
                f" reference point at {coord_word}"
                f" {describe_arraylike(self.relative_to_position, precision=precision)}."
                + " " + common_instruction
            )
        if self.mode_flag == "relative_to_pair_motif":
            relative_motif_kwargs.update({"style": "index"})
            return (
                f"add [{motif.describe(**motif_kwargs)}] to the structure,"
                f" with its centroid located on the line between"
                f" atoms in [{self.relative_to_motif.describe(**relative_motif_kwargs)}], at"
                f" {self.relative_shift:.{precision}f} angstroms away from the atom indexed"
                f" {self.relative_to_motif.indices[self.relative_atom_index]}."
                + " " + common_instruction
            )
        if self.mode_flag == "relative_to_regular_motif":
            return (
                f"add [{motif.describe(**motif_kwargs)}] to the structure,"
                f" with its centroid shifted in {coord_word} by"
                f" {describe_arraylike(self.relative_shift, precision=precision)} relative to the"
                f" centroid of [{self.relative_to_motif.describe(**relative_motif_kwargs)}]."
                + " " + common_instruction
            )
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}")
