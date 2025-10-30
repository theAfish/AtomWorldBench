"""Implement Translate action."""
from typing import Optional

from ase import Atoms
from numpy.typing import ArrayLike
import numpy as np
import inspect

from .base import BaseMotifAction
from ...motifs.base import BaseMotif

from ....utils.coord_utils import check_coordinates_shape
from ....utils.atoms_utils import merge_atoms
from ....utils.description_utils import describe_arraylike
from ....common.globals import DEFAULT_FLOAT_TO_STRING_PRECISION
from ....common.registry import register

from .utils import _must_be_non_bond_site_collection_motif


def _check_translation_vector(v, mode_flag):
    if mode_flag == "relative_to_self":
        v = check_coordinates_shape(
                v, "relative_to_position", expected_1d=True, allow_none=True
            )
    elif mode_flag in ["relative_to_motif", "relative_to_position"]:
        if not isinstance(v, (int, float)):
            raise ValueError(
                "translation_vector must be a scalar distance when"
                f" mode_flag is {mode_flag}."
            )
    return v


# Can only be called "translate-motif" as "translate" may conflict with TranslateStructureAction.
@register(BaseMotifAction, ["translate", "translate-motif"])
class TranslateMotifAction(BaseMotifAction):
    """Action to translate a motif in the structure."""
    kwargs_formatting_functions = {
        "to_position":
            lambda x: check_coordinates_shape(
                x, "to_position", expected_1d=True, allow_none=True
            ),
        "relative_to_position":
            lambda x: check_coordinates_shape(
                x, "relative_to_position", expected_1d=True, allow_none=True
            ),
        "translation_vector": _check_translation_vector,
        "operated_motif": _must_be_non_bond_site_collection_motif,
        "relative_to_motif": _must_be_non_bond_site_collection_motif,
    }
    mode_definitions = {
        # operated_motif, operated_atoms and translation_vector are always required.
        # position_fractional does not need to be checked.
        "_excluded": [
            "operated_motif",
            "position_fractional"
        ],
        "absolute": {"to_position": None},
        "relative_to_position": {
            "relative_to_position": None,
            "translation_vector": None,
        },
        "relative_to_motif": {
            "relative_to_motif": None,
            "translation_vector": None,
        },
        "relative_to_self": {
            "translation_vector": None,
        }
    }

    def __init__(
            self,
            operated_motif: BaseMotif,
            # Optional Parameters.
            relative_to_motif: Optional[BaseMotif] = None,
            to_position: Optional[ArrayLike] = None,
            relative_to_position: Optional[ArrayLike] = None,
            translation_vector: Optional[ArrayLike|int|float] = None,
            position_fractional: bool = False,
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
                In this mode, the translation_vector must be a scalar distance, positive or
                negative, indicating the distance to move away from or towards the
                reference position along the line connecting the motif centroid and the
                reference position.
            3, "relative_to_motif": translation to a position relative to a specified motif.
                In this mode, `relative_to_motif` and `translation_vector` are required.
                No other parameters are allowed except `position_fractional`.
                In this mode, the translation_vector must be a scalar distance, positive or
                negative, indicating the distance to move away from or towards the
                reference motif along the line connecting the motif centroid and the
                reference motif centroid.
            4, "relative_to_self": direct translation of the provided motif to the
                `execute` method by a translation vector. No need for relative reference
                position or motif. In this mode, `self_relative` and `translation_vector`
                are required. No other parameters are allowed except `position_fractional`.
                In this mode, the translation_vector is a full 3D vector.

        Args:
            operated_motif (BaseMotif): The motif to be translated.
            to_position (ArrayLike, optional): The position to translate the motif to.
                Turns on the absolute mode of the action.
            relative_to_position (ArrayLike, optional): The position to translate
                the motif relative to.
            translation_vector (ArrayLike or int or float, optional): The vector by which to translate
                the motif.
                In "relative_to_self" mode, this is a full 3D vector.
                In "relative_to_position" and "relative_to_motif" modes, this is a scalar distance,
                positive or negative, indicating the distance to move away from or towards the
                reference position or motif along the line connecting the motif centroid and the
                reference position or motif centroid. In these modes, unit is angstroms.
            position_fractional (bool, optional): If True, the positions and translation
                vector are in fractional coordinates. If False, they are in Cartesian
                coordinates. This will also affect the description style of the action.
                Default is False.
            relative_to_motif (BaseMotif, optional): A motif that the action is taken
                relative to. This can be used to define the context of the action.
        """
        # Just for linting.
        self.to_position = None
        self.position_fractional = None
        self.translation_vector=None
        self.relative_to_position=None
        super().__init__(
            operated_motif=operated_motif,
            relative_to_motif=relative_to_motif,
            to_position=to_position,
            relative_to_position=relative_to_position,
            translation_vector=translation_vector,
            position_fractional=position_fractional,
        )

    def __post_init__(self):
        """Post-initialization to ensure the action is valid."""
        self._check_operated_motif_in_atoms()
        self._check_relative_motif_in_atoms()
        # Check centroid not overlapping for relative_to_motif mode.
        if self.relative_to_motif is not None:
            operated_centroid = self.operated_motif.get_centroid(fractional=False)
            relative_centroid = self.relative_to_motif.get_centroid(fractional=False)
            if np.allclose(operated_centroid, relative_centroid):
                raise ValueError(
                    "The centroids of the operated motif and the relative motif are"
                    " at the same position."
                    " Cannot determine translation direction."
                )
            distance = np.linalg.norm(operated_centroid - relative_centroid)
            if self.translation_vector < 0 and abs(self.translation_vector) > distance:
                raise ValueError(
                    "The translation distance to move inward"
                    " is larger than the distance between"
                    " the operated motif and the relative motif."
                    " This will cause overshoot."
                )
        # Relative position can not be the same as operated motif centroid.
        if self.relative_to_position is not None:
            operated_centroid = self.operated_motif.get_centroid(fractional=False)
            relative_position = (
                self.relative_to_position
                @ self.operated_atoms.cell.complete()
                if self.position_fractional else self.relative_to_position
            )
            if np.allclose(operated_centroid, relative_position):
                raise ValueError(
                    "The centroid of the operated motif and the reference point are"
                    " at the same position."
                    " Cannot determine translation direction."
                )
            distance = np.linalg.norm(operated_centroid - relative_position)
            if self.translation_vector < 0 and abs(self.translation_vector) > distance:
                raise ValueError(
                    "The translation distance to move inward"
                    " is larger than the distance between"
                    " the operated motif and the reference point."
                    " This will cause overshoot."
                )


    def _get_translation_vector(self) -> ArrayLike:
        """Get the translation vector based on the action parameters.

        Always return cartesian coordinates.
        """
        if self.mode_flag == "absolute":
            to_position = self.to_position if not self.position_fractional else (
                self.to_position @ self.operated_atoms.cell.complete()
            )
            return (
                    to_position -
                    self.operated_motif.get_centroid(fractional=False)
            )

        if self.mode_flag == "relative_to_motif":
            relative_centroid = self.relative_to_motif.get_centroid(fractional=False)
            operated_centroid = self.operated_motif.get_centroid(fractional=False)
            direction_vector = relative_centroid - operated_centroid
            unit = direction_vector / np.linalg.norm(direction_vector)
            return -unit * self.translation_vector  # Positive: away from relative motif.

        if self.mode_flag == "relative_to_position":
            if not self.position_fractional:
                relative_centroid = self.relative_to_position
            else:
                relative_centroid = (
                    self.relative_to_position
                    @ self.operated_atoms.cell.complete()
                )
            operated_centroid = self.operated_motif.get_centroid(fractional=False)
            direction_vector = relative_centroid - operated_centroid
            unit = direction_vector / np.linalg.norm(direction_vector)
            return -unit * self.translation_vector  # Positive: away from relative motif.

        if self.mode_flag == "relative_to_self":
            return self.translation_vector if not self.position_fractional else (
                self.translation_vector @ self.operated_atoms.cell.complete()
            )

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

        ## Remove the motif from the structure, then add translated motif back.
        # __post_init__ ensures that the motif is in the atoms and has indices.
        indices = self.operated_motif.indices
        other_indices = np.sort(np.setdiff1d(
            np.arange(len(self.operated_atoms), dtype=int), indices, assume_unique=True
        )).tolist()
        motif_atoms = self.operated_motif.get_atoms().copy()
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
                relative motif will be overwritten by this parameter.
            motif_desc_kwargs (dict, optional): Additional keyword arguments for the motif description.
            relative_motif_desc_kwargs (dict, optional): Additional keyword arguments for the relative
                motif description.

        Returns:
            str: A description of the action.
        """
        motif_desc_kwargs = motif_desc_kwargs or {}
        relative_motif_desc_kwargs = relative_motif_desc_kwargs or {}

        # Update motif description kwargs.
        motif_desc_params = inspect.signature(self.operated_motif.describe).parameters
        relative_motif_desc_params = inspect.signature(
            self.relative_to_motif.describe
        ).parameters if self.relative_to_motif is not None else {}
        # Never use addition mode for rotation.
        if "precision" in motif_desc_params:
            motif_desc_kwargs["precision"] = precision
        if "precision" in relative_motif_desc_params:
            relative_motif_desc_kwargs["precision"] = precision

        if self.position_fractional:
            coord_word = "fractional coordinates"
        else:
            coord_word = "cartesian coordinates"

        # A common instruction to prevent shuffling indices.
        common_instruction = (
            "update atom positions only, do not change their order in structure."
        )

        if self.mode_flag == "absolute":
            return (
                    f"translate [{self.operated_motif.describe(**motif_desc_kwargs)}]"
                    f" so as to relocate its centroid at {coord_word}"
                    f" {describe_arraylike(self.to_position, precision=precision)}."
                    + " " + common_instruction
            )
        if self.mode_flag == "relative_to_position":
            move_word = "away from" if self.translation_vector >= 0 else "towards"
            return (
                f"translate [{self.operated_motif.describe(**motif_desc_kwargs)}]"
                f" so as to move its centroid {self.translation_vector:.{precision}f}"
                f" angstroms {move_word} a reference point at {coord_word}"
                f" {describe_arraylike(self.relative_to_position, precision=precision)}."
                + " " + common_instruction
            )
        if self.mode_flag == "relative_to_motif":
            move_word = "away from" if self.translation_vector >= 0 else "towards"
            return (
                f"translate [{self.operated_motif.describe(**motif_desc_kwargs)}]"
                f" so as to move its centroid {self.translation_vector:.{precision}f}"
                f" angstroms {move_word} the centroid of"
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
