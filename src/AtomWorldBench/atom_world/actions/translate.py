"""Implement Translate action."""
from typing import Optional, Tuple

from ase import Atoms
from numpy.typing import ArrayLike
import numpy as np
import inspect

from .base import BaseAction
from ..motifs.base import BaseMotif

from ...utils.coord_utils import check_coordinates_shape
from ...utils.atoms_utils import merge_atoms
from ...utils.description_utils import describe_arraylike
from ...globals import DEFAULT_FLOAT_TO_STRING_PRECISION


class TranslateMotifAction(BaseAction):
    """Action to translate a motif in the structure."""
    kwargs_formating_functions = {
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
        # operated_motif, operated_atoms and translation_vector are always required.
        # position_fractional does not need to be checked.
        "_excluded": [
            "operated_motif", "operated_atoms",
            "position_fractional"
        ],
        "absolute": {"to_position": None},
        "relative_to_position": {
            "relative_to_position": None,
            "translation_vector": None,
        },
        "relative_to_motif": {
            "relative_to_motif": None,
            "relative_style": (
                lambda s: s == "centroid_distance",
                "relative_style must be centroid_distance for relative_to_motif translation mode."
            ),
            "translation_vector": None,
        },
        "relative_to_self": {
            "relative_style": (
                lambda s: s == "self",
                "relative_style must be self for relative_to_self translation mode."
            ),
            "translation_vector": None,
        }
    }

    def __init__(
            self,
            operated_motif: BaseMotif,
            operated_atoms: Atoms,
            # Optional Parameters.
            relative_to_motif: Optional[BaseMotif] = None,
            to_position: Optional[ArrayLike] = None,
            relative_to_position: Optional[ArrayLike] = None,
            translation_vector: Optional[ArrayLike] = None,
            position_fractional: bool = False,
            relative_style: str = None,
    ):
        """Initialize the TranslateMotifAction with a relative motif and style.

        `operated_motif`, `operated_atoms` and `translation_vector` are always required.
        For the rest of parameters currently, allows 4 modes of operation:
            1, "absolute": translation directly to a specified position. In this mode,
                the `to_position` parameter is required. No other parameters are allowed
                except `position_fractional`.
            2, "relative_to_position": translation to a position relative to a specified
                position. In this mode, `relative_to_position` and `translation_vector`
                are required. No other parameters are allowed except `position_fractional`.
            3, "relative_to_motif": translation to a position relative to a specified motif.
                In this mode, `relative_to_motif`, `translation_vector` and
                `relative_style` == "centroid_distance" are required.
                No other parameters are allowed except `position_fractional`.
            4, "relative_to_self": direct translation of the provided motif to the
                `execute` method by a translation vector. No need for relative reference
                position or motif. In this mode, `self_relative` and `translation_vector`
                are required. No other parameters are allowed except `position_fractional`.

        Args:
            operated_motif (BaseMotif): The motif to be translated.
            operated_atoms (Atoms): The structure from which the motif is to be translated.
            to_position (ArrayLike, optional): The position to translate the motif to.
                Turns on the absolute mode of the action.
            relative_to_position (ArrayLike, optional): The position to translate
                the motif relative to.
            translation_vector (ArrayLike, optional): The vector by which to translate
                the motif with respect to the `relative_to_position` or relative motif.
            position_fractional (bool, optional): If True, the positions and translation
                vector are in fractional coordinates. If False, they are in Cartesian
                coordinates. This will also affect the description style of the action.
                Default is False.
            relative_to_motif (BaseMotif, optional): A motif that the action is taken
                relative to. This can be used to define the context of the action.
            relative_style (str, optional): The style to determine relative action.
                For example, an action can be relative to a motif's centroid in distance.
                See `allowed_relative_styles` for the list of allowed styles. If None,
                will use the first style in `allowed_relative_styles`.
        """
        super().__init__(
            operated_motif=operated_motif,
            operated_atoms=operated_atoms,
            relative_to_motif=relative_to_motif,
            to_position=to_position,
            relative_to_position=relative_to_position,
            translation_vector=translation_vector,
            position_fractional=position_fractional,
            relative_style=relative_style,
        )

    def __post_init__(self):
        """Post-initialization to ensure the action is valid."""
        self.__check_operated_motif_compatibility()
        self.__check_operated_motif_in_atoms()
        self.__check_relative_motif_in_atoms()

    def _get_translation_vector(self) -> ArrayLike:
        """Get the translation vector based on the action parameters."""
        if self.mode_flag == "absolute":
            return (
                    self.to_position -
                    self.operated_motif.get_centroid(fractional=self.position_fractional)
            )

        if self.mode_flag == "relative_to_motif":
            return (
                    self.relative_to_motif.get_centroid(fractional=self.position_fractional)
                    + self.translation_vector
                    - self.operated_motif.get_centroid(fractional=self.position_fractional)
            )

        if self.mode_flag == "relative_to_position":
            return (
                    self.relative_to_position
                    + self.translation_vector
                    - self.operated_motif.get_centroid(fractional=self.position_fractional)
            )

        if self.mode_flag == "relative_to_self":
            return self.translation_vector

        raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}.")

    def execute(self) -> Atoms:
        """Execute the action to translate the motif in the structure.

        Translates the motif in the structure based on the action parameters.
        Order of atoms in the structure is preserved, but the motif is translated.
        Returns:
            Atoms: The modified structure with the motif translated.
        """
        # Atoms.translate only accepts Cartesian coordinates, so we need to convert.
        translation_vector = self._get_translation_vector()
        if self.position_fractional:
            translation_vector = translation_vector @ self.operated_atoms.cell.complete()

        ## Remove the motif from the structure, then add translated motif back.
        # __post_init__ ensures that the motif is in the atoms and has indices.
        indices = self.operated_motif.indices
        other_indices = np.sort(np.setdiff1d(
            np.arange(len(self.operated_atoms), dtype=int), indices, assume_unique=True
        )).tolist()
        motif_atoms = self.operated_motif.get_atoms()
        motif_atoms.translate(translation_vector)
        return merge_atoms(
            [self.operated_atoms[other_indices], motif_atoms],
            [other_indices, indices]
        )

    def describe(
            self,
            precision: int = DEFAULT_FLOAT_TO_STRING_PRECISION,
            motif_desc_kwargs: Optional[dict] = None,
            relative_motif_desc_kwargs: Optional[dict] = None,
    ) -> str:
        """Describe the action to translate a motif.

         Note that motif and relative motif description styles are not affected by the action's
        `position_fractional` attribute.
        Args:
            precision (int): The precision for formatting numerical values in the description in decimals.
                Default is set in `globals.py`, typically 4.
                Note that the precision in the description of the operated motif and the
                relative motif is controlled by the `motif_desc_kwargs` and
                `relative_motif_desc_kwargs` parameters, respectively, not by this parameter!
            motif_desc_kwargs (dict, optional): Additional keyword arguments for the motif description.
            relative_motif_desc_kwargs (dict, optional): Additional keyword arguments for the relative
                motif description.

        Returns:
            str: A description of the action.
        """
        motif_desc_kwargs = motif_desc_kwargs or {}
        relative_motif_desc_kwargs = relative_motif_desc_kwargs or {}

        # Update motif description kwargs.
        motif_desc_params = inspect.signature(self.motif.describe).parameters
        relative_motif_desc_params = inspect.signature(
            self.relative_to_motif.describe
        ).parameters if self.relative_to_motif is not None else {}
        # Never use addition mode for rotation.
        if "is_addition" in motif_desc_params:
            motif_desc_kwargs["is_addition"] = False
        if "is_addition" in relative_motif_desc_params:
            relative_motif_desc_kwargs["is_addition"] = False

        if self.position_fractional:
            coord_word = "fractional coordinates"
        else:
            coord_word = "cartesian coordinates"

        # A common instruction to prevent shuffling indices.
        common_instruction = (
            "update atom coordinates only, do not change their order in structure."
        )

        if self.mode_flag == "absolute":
            return (
                    f"translate [{self.operated_motif.describe(**motif_desc_kwargs)}]"
                    f" so as to relocate its centroid at {coord_word}"
                    f" {describe_arraylike(self.to_position, precision=precision)}."
                    + " " + common_instruction
            )
        if self.mode_flag == "relative_to_position":
            return (
                f"translate [{self.operated_motif.describe(**motif_desc_kwargs)}]"
                f" so as to relocate its centroid at {coord_word}"
                f" {describe_arraylike(self.translation_vector, precision=precision)}"
                f" shifted from a reference point at {coord_word}"
                f" {describe_arraylike(self.relative_to_position, precision=precision)}."
                + " " + common_instruction
            )
        if self.mode_flag == "relative_to_motif":
            return (
                f"translate [{self.operated_motif.describe(**motif_desc_kwargs)}]"
                f" so as to relocate its centroid at {coord_word}"
                f" {describe_arraylike(self.translation_vector, precision=precision)}"
                f" shifted from the centroid of"
                f" [{self.relative_to_motif.describe(**relative_motif_desc_kwargs)}]."
                + " " + common_instruction
            )
        if self.mode_flag == "relative_to_self":
            return (
                f"translate [{self.operated_motif.describe(**motif_desc_kwargs)}] by"
                f" {describe_arraylike(self.translation_vector, precision=precision)}"
                f" in {coord_word}."
                + " " + common_instruction
            )
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}")
