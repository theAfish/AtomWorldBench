"""Implement Translate action."""
from typing import Optional, Tuple

from ase import Atoms
from numpy.typing import ArrayLike
import numpy as np

from .base import BaseAction
from ..motifs.base import BaseMotif

from ...utils.coord_utils import check_coordinates_shape
from ...utils.atoms_utils import merge_atoms
from ...utils.description_utils import describe_arraylike
from ...globals import DEFAULT_FLOAT_TO_STRING_PRECISION


class TranslateMotifAction(BaseAction):
    """Action to translate a motif in the structure."""
    allowed_relative_styles = [
        "centroid_distance",  # Translate relative to the centroid of the relative motif.
    ]
    kwargs_and_formating_functions = {
        "to_position":
            lambda x: check_coordinates_shape(
                x, "to_position", expected_1d=True, allow_none=True
            ),
        "relative_to_position":
            lambda x: check_coordinates_shape(
                x, "relative_to_position", expected_1d=True, allow_none=True
            ),
        "translation_vector":
            lambda x: check_coordinates_shape(
                x, "translation_vector", expected_1d=True, allow_none=True
            ),
    }
    mode_definitions = {
        "_excluded": ["position_fractional"],
        "absolute": {"to_position": None},
        "relative_to_position": {
            "relative_to_position": None, "translation_vector": None
        },
        "relative_to_motif": {
            "relative_to_motif": None, "translation_vector": None,
            "relative_style": None
        },
        "relative_to_self": {
            "self_relative": (
                lambda x: x is True,
                "self_relative must be True for relative_to_self translation mode."
            ),
            "translation_vector": None
        }
    }

    def __init__(
            self,
            to_position: Optional[ArrayLike] = None,
            relative_to_position: Optional[ArrayLike] = None,
            translation_vector: Optional[ArrayLike] = None,
            position_fractional: bool = True,
            relative_style: str = None,
            relative_to_motif: Optional[BaseMotif] = None,
            self_relative: Optional[bool] = None,
    ):
        """Initialize the TranslateMotifAction with a relative motif and style.

        Currently, allows 4 modes of operation:
            1, "absolute": translation directly to a specified position. In this mode,
                the `to_position` parameter is required. No other parameters are allowed
                except `position_fractional`.
            2, "relative_to_position": translation to a position relative to a specified
                position. In this mode, `relative_to_position` and `translation_vector`
                are required. No other parameters are allowed except `position_fractional`.
            3, "relative_to_motif": translation to a position relative to a specified motif.
                In this mode, `relative_to_motif`, `translation_vector` and `relative_style`
                are required. No other parameters are allowed except `position_fractional`.
            4, "relative_to_self": direct translation of the provided motif to the
                `execute` method by a translation vector. No need for relative reference
                position or motif. In this mode, `self_relative` and `translation_vector`
                are required. No other parameters are allowed except `position_fractional`.

        Args:
            to_position (ArrayLike, optional): The position to translate the motif to.
                Turns on the absolute mode of the action.
            relative_to_position (ArrayLike, optional): The position to translate
                the motif relative to.
            translation_vector (ArrayLike, optional): The vector by which to translate
                the motif with respect to the `relative_to_position` or relative motif.
            position_fractional (bool, optional): If True, the positions are in fractional
                coordinates. If False, the positions are in Cartesian coordinates.
                This will also affect the description style of the action.
                Default is True.
            relative_to_motif (BaseMotif, optional): A motif that the action is taken
                relative to. This can be used to define the context of the action.
            relative_style (str, optional): The style to determine relative action.
                For example, an action can be relative to a motif's centroid in distance.
                See `allowed_relative_styles` for the list of allowed styles. If None,
                will use the first style in `allowed_relative_styles`.
            self_relative (bool, optional): If True, the action is relative to the motif itself
                as passed into `execute`.
        """
        # Static declaration for IDE linting.
        self.to_position = None
        self.relative_to_position = None
        self.translation_vector = None
        self.position_fractional = None
        self.relative_style = None
        self.relative_to_motif = None
        self.self_relative = None

        super().__init__(
            to_position=to_position,
            relative_to_position=relative_to_position,
            translation_vector=translation_vector,
            position_fractional=position_fractional,
            relative_style=relative_style,
            relative_to_motif=relative_to_motif,
            self_relative=self_relative,
        )

    def _check_compatibility(self, atoms: Atoms, motif: BaseMotif) -> Tuple[bool, str]:
        """Check if the motif can be translated in the structure."""
        # Check if the operated motif is in the structure.
        indices = motif.find_indices_in_atoms(atoms, modify_indices_in_place=True)
        if indices is not None:
            return True, ""
        return False, "Motif not found in the structure."

    def _get_translation_vector(
        self, motif: BaseMotif
    ) -> ArrayLike:
        """Get the translation vector based on the action parameters."""
        if self.mode_flag == "absolute":
            return self.to_position - motif.get_centroid(fractional=self.position_fractional)

        if self.mode_flag == "relative_to_motif":
            return (
                    self.relative_to_motif.get_centroid(fractional=self.position_fractional)
                    + self.translation_vector
                    - motif.get_centroid(fractional=self.position_fractional)
            )

        if self.mode_flag == "relative_to_position":
            return (
                    self.relative_to_position
                    + self.translation_vector
                    - motif.get_centroid(fractional=self.position_fractional)
            )

        if self.mode_flag == "relative_to_self":
            return self.translation_vector

        raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}.")

    def _execute(self, atoms: Atoms, motif: BaseMotif) -> Atoms:
        """Execute the action to translate the motif in the structure.

        Translates the motif in the structure based on the action parameters.
        Order of atoms in the structure is preserved, but the motif is translated.
        Args:
            atoms (Atoms): The structure to operate on.
            motif (BaseMotif): The motif to translate.

        Returns:
            Atoms: The modified structure with the motif translated.
        """
        # Atoms.translate only accepts Cartesian coordinates, so we need to convert.
        translation_vector = self._get_translation_vector(motif)
        if self.position_fractional:
            translation_vector = translation_vector @ atoms.cell.complete()

        ## Remove the motif from the structure, then add translated motif back.
        indices = motif.find_indices_in_atoms(atoms, modify_indices_in_place=True)
        other_indices = np.setdiff1d(
            np.arange(len(atoms), dtype=int), indices, assume_unique=True
        ).tolist()
        motif_atoms = motif.get_atoms()
        motif_atoms.translate(translation_vector)
        return merge_atoms(
            [atoms[other_indices], motif_atoms],
            [other_indices, indices]
        )

    def describe(
            self,
            motif: BaseMotif,
            precision: int = DEFAULT_FLOAT_TO_STRING_PRECISION,
            motif_kwargs: Optional[dict] = None,
            relative_motif_kwargs: Optional[dict] = None,
    ) -> str:
        """Describe the action to translate a motif.

         Note that motif and relative motif description styles are not affected by the action's
        `position_fractional` attribute.
        Args:
            motif (BaseMotif): The motif being translated.
            precision (int): The precision for formatting numerical values in the description in decimals.
                Default is set in `globals.py`, typically 4.
                Will overwrite motif and relative motif description precision settings.
            motif_kwargs (dict, optional): Additional keyword arguments for the motif description.
            relative_motif_kwargs (dict, optional): Additional keyword arguments for the relative
                motif description.

        Returns:
            str: A description of the action.
        """
        motif_kwargs = motif_kwargs or {}
        relative_motif_kwargs = relative_motif_kwargs or {}

        motif_kwargs.update({"precision": precision, "is_addition": False})
        relative_motif_kwargs.update({"precision": precision, "is_addition": False})

        if self.position_fractional:
            coord_word = "fractional coordinates"
        else:
            coord_word = "cartesian coordinates"

        # A common instruction to prevent shuffling indices.
        common_instruction = "modify coordinates only, do not change the order of atoms in structure."

        if self.mode_flag == "absolute":
            return (
                    f"translate [{motif.describe(**motif_kwargs)}]"
                    f" so as to relocate its centroid at {coord_word}"
                    f" {describe_arraylike(self.to_position, precision=precision)}."
                    + " " + common_instruction
            )
        if self.mode_flag == "relative_to_position":
            return (
                f"translate [{motif.describe(**motif_kwargs)}] so as to relocate its centroid"
                f" at {coord_word} {describe_arraylike(self.translation_vector, precision=precision)}"
                f" relative to a reference point at {coord_word}"
                f" {describe_arraylike(self.relative_to_position, precision=precision)}."
                + " " + common_instruction
            )
        if self.mode_flag == "relative_to_motif":
            return (
                f"translate [{motif.describe(**motif_kwargs)}] so as to relocate its centroid"
                f" at {coord_word} {describe_arraylike(self.translation_vector, precision=precision)}"
                f" relative to the centroid of [{self.relative_to_motif.describe(**relative_motif_kwargs)}]."
                + " " + common_instruction
            )
        if self.mode_flag == "relative_to_self":
            return (
                f"translate [{motif.describe(**motif_kwargs)}] in {coord_word} by"
                f" {describe_arraylike(self.translation_vector, precision=precision)}."
                + " " + common_instruction
            )
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}")
