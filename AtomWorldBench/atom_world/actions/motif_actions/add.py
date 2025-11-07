"""Implementation of actions that add a motif to a structure."""
from typing import Optional
import inspect

import numpy as np
from ase import Atoms
from numpy.typing import ArrayLike

from .base import BaseMotifAction
from .utils import get_random_motif
from ...motifs.base import BaseMotif
from ...motifs.site_collections.base import BaseSiteCollectionMotif
from ....utils.description_utils import describe_arraylike
from ....utils.coord_utils import check_coordinates_shape

from ....common.globals import DEFAULT_FLOAT_TO_STRING_PRECISION
from ....common.registry import register

from .utils import _must_be_non_bond_site_collection_motif


def _check_relative_shift(x):
    if isinstance(x, (int, float)):
        return float(x)
    else:
        return check_coordinates_shape(
            x, "relative_shift", expected_1d=True, allow_none=True
        )

def _check_relative_motif_compatibility(m, mode_flag):
    """Check if the relative motif is compatible with the action."""
    if mode_flag == "relative_to_motif_centroid":
        if not isinstance(m, BaseSiteCollectionMotif):
            raise ValueError(
                "Relative to motif must be a site collection motif for"
                " relative_to_motif_centroid mode."
            )
    if mode_flag == "relative_to_pair_motif":
        if not (isinstance(m, BaseSiteCollectionMotif) and len(m) == 2):
            raise ValueError(
                "Only pair motifs are allowed for relative_to_pair_motif mode."
            )
    return m

@register(BaseMotifAction, ["add", "add-motif"])
class AddMotifAction(BaseMotifAction):
    """Action to add a motif to a structure.

    This action defines how to add a motif to a given structure, including the
    description of the action and the execution logic.
    """
    kwargs_formatting_functions = {
        "at_position":
            lambda x: check_coordinates_shape(
                x, "at_position", expected_1d=True, allow_none=True
            ),
        "relative_to_position":
            lambda x: check_coordinates_shape(
                x, "relative_to_position", expected_1d=True, allow_none=True
            ),
        "relative_shift": _check_relative_shift,
        "operated_motif": _must_be_non_bond_site_collection_motif,
        "relative_to_motif": _check_relative_motif_compatibility,
    }
    mode_definitions = {
        # Operated_motif and operated_atoms are always required.
        # position_fractional doesn't need to be checked.
        "_excluded": ["position_fractional", "operated_motif", "operated_atoms"],
        "absolute": {"at_position": None},
        "relative_to_position": {
            "relative_to_position": None,
            "relative_shift": None,  # Shape will be checked later.
        },
        "relative_to_motif_centroid": {
            "relative_to_motif": None,
            "relative_shift": None,
            "relative_style": (
                lambda s: s  == "centroid_distance",
                "Relative style must be centroid_distance for"
                " relative_to_motif_centroid mode."
            ),
        },
        "relative_to_pair_motif": {
            "relative_to_motif": (
                lambda m: isinstance(m, BaseSiteCollectionMotif) and len(m) == 2,
                "Only pair motifs are allowed for relative_to_pair_motif mode."
            ),
            "relative_shift": (
                lambda n: isinstance(n, (int, float)),
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
    mode_probabilities = {
        "absolute": 0.3,
        "relative_to_position": 0.2,
        "relative_to_motif_centroid": 0.2,
        "relative_to_pair_motif": 0.3,
    }

    # Deprecate BaseAction's operated_atoms property.
    operated_atoms = None
    def __init__(
            self,
            operated_motif: BaseMotif,
            operated_atoms: Atoms,
            relative_to_motif: Optional[BaseMotif] = None,
            at_position: Optional[ArrayLike] = None,
            relative_to_position: Optional[ArrayLike] = None,
            position_fractional: bool = False,
            relative_style: Optional[str] = None,
            relative_shift: Optional[ArrayLike | float] = None,
            relative_atom_index: Optional[int] = None,
    ):
        """Initialize the AddMotifAction with optional parameters.

        `operated_motif` and `operated_atoms` are always required.
        For the rest of parameters, currently, allows 4 modes of operation:
            1, `absolute`: Add the motif at a specified position in the structure.
                In this mode, only `at_position` is provided, no other parameters
                should be given except position_fractional.
            2, `relative_to_position`: Add the motif relative to a specified position.
                In this mode, provide `relative_to_position` and `relative_shift`.
                No other parameters should be given except position_fractional.
            3, `relative_to_motif_centroid`: Add the motif relative to a specified motif's
                centroid. In this mode, provide `relative_to_motif`, `relative_shift`,
                and `relative_style`=="centroid_distance".
                No other parameters should be given except
                position_fractional. Relative motif can be any motif that supports centroid
                relative style.
            4, `relative_to_pair_motif`: Add the motif on a pair motif's line. In this
                mode, provide `relative_to_motif`, `relative_shift` (as a float distance
                in angstroms),
                `relative_style` == "position_in_line", and `relative_atom_index`.
                No other parameters should be given except position_fractional.
        Args:
            operated_motif (BaseMotif): The motif to be added to the structure.
            operated_atoms (Atoms): The Atoms object that the motif is added to.
            relative_to_motif (BaseMotif): The motif to which the operated motif is added
                relative to.
            at_position (ArrayLike, optional): The position where the motif is added.
                If provided, it overrides all relative parameters.
            relative_to_position (ArrayLike, optional): The position to which the motif
                is added relative to.
            position_fractional (bool, optional):
                Whether all positions and `relative_shift` are fractional. If False,
                will be cartesian. This will also affect the description style of the action.
                Default is False.
            relative_style (str, optional): The style to determine relative action with
                respect to `relative_to_motif`.
            relative_shift (ArrayLike or float, optional):
                 A vector or float distance defining the relative position.
            relative_atom_index (int, optional):
                 The index of the atom in the relative motif to insert atom at `relative_shift`
                 distance, if relative_style is `position_in_line`. Must be provided if working
                 in `position_in_line` mode.
        """
        # Just for Linting purpose.
        self.position_fractional = None
        self.operated_atoms = None
        self.at_position = None
        self.relative_to_position = None
        self.relative_to_motif = None
        self.relative_style = None
        self.relative_shift = None
        self.relative_atom_index = None
        super().__init__(
            operated_motif=operated_motif,
            operated_atoms=operated_atoms,
            relative_to_motif=relative_to_motif,
            at_position=at_position,
            relative_to_position=relative_to_position,
            position_fractional=position_fractional,
            relative_style=relative_style,
            relative_shift=relative_shift,
            relative_atom_index=relative_atom_index,
        )

    def __post_init__(self):
        # AddAction does not need to check operated_motif existence in operated_atoms.
        self._check_relative_motif_in_atoms()
        # Operated motif must be in additive mode.
        if not self.operated_motif.is_additive:
            raise ValueError("Inserted motif must be in additive mode.")

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
            centroid = self.relative_to_motif.cart_coords[self.relative_atom_index]
            ref_position = self.relative_to_motif.cart_coords[1 - self.relative_atom_index]
            bond_norm_vec = (ref_position - centroid) / np.linalg.norm(ref_position - centroid)
            relative_shift = self.relative_shift * bond_norm_vec
        # Insert at distance to relative motif centroid.
        elif self.mode_flag == "relative_to_motif_centroid":
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

    def execute(self) -> Atoms:
        """Execute the action on the structure to generate the ground truth structure.

        Added motif will always be appended to the end of the structure.
        Returns:
            Atoms: The modified structure with the motif added.
        """
        insert_position = self._compute_insert_cart_position(self.operated_atoms)
        displacement = insert_position - self.operated_motif.get_centroid(fractional=False)
        operated_motif_atoms = self.operated_motif.get_atoms()
        operated_motif_atoms.translate(displacement)
        new_atoms = self.operated_atoms.copy()
        new_atoms += operated_motif_atoms
        return new_atoms

    def describe(
            self,
            precision: int = DEFAULT_FLOAT_TO_STRING_PRECISION,
            motif_desc_kwargs: Optional[dict] = None,
            relative_motif_desc_kwargs: Optional[dict] = None,
    ) -> str:
        """Describe the action for LLM prompting.

        Args:
            precision (int): The precision for formatting numerical values in the description
                in decimals. Default is set in `globals.py`, typically 4.
                Note that the precision in the description of the operated motif and the
                relative motif in the `motif_desc_kwargs` and `relative_motif_desc_kwargs`
                 parameters will be overwritten by this parameter!
            motif_desc_kwargs (dict, optional): Additional keyword arguments for the motif
                description. See motif.describe() for available options.
            relative_motif_desc_kwargs (dict, optional): Additional keyword arguments for
                the relative motif description. See motif.describe() for available options.
            Note that motif and relative motif description styles are not affected by the
            action's `position_fractional` attribute.
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
        # Use addition mode as site motif needs different short description.
        # when being added.
        if "precision" in motif_desc_params:
            motif_desc_kwargs["precision"] = precision
        if "precision" in relative_motif_desc_params:
            relative_motif_desc_kwargs["precision"] = precision

        if self.position_fractional:
            coord_word = "fractional coordinates"
        else:
            coord_word = "cartesian coordinates"

        # Common instruction to guarantee order of addition.
        common_instruction = (
            "newly added motif should be appended to the end of the structure,"
            " in the order as described."
        )

        motif = self.operated_motif
        rel_motif = self.relative_to_motif
        if self.mode_flag == "absolute":
            return (
                    f"add [{motif.describe(**motif_desc_kwargs)}] to the structure,"
                    f" with its centroid located at {coord_word}"
                    f" {describe_arraylike(self.at_position, precision=precision)}."
                    + " " + common_instruction
            )
        if self.mode_flag == "relative_to_position":
            return (
                f"add [{motif.describe(**motif_desc_kwargs)}] to the structure,"
                f" with its centroid shifted in {coord_word} by"
                f" {describe_arraylike(self.relative_shift, precision=precision)} relative to a"
                f" reference point at {coord_word}"
                f" {describe_arraylike(self.relative_to_position, precision=precision)}."
                + " " + common_instruction
            )
        if self.mode_flag == "relative_to_pair_motif":
            # Must use index description style for pair motifs.
            relative_motif_desc_kwargs.update({"style": "index"})
            return (
                f"add [{motif.describe(**motif_desc_kwargs)}] to the structure,"
                f" with its centroid located on the line between"
                f" atoms in [{rel_motif.describe(**relative_motif_desc_kwargs)}], at"
                f" {self.relative_shift:.{precision}f} angstroms away from the atom indexed"
                f" {rel_motif.indices[self.relative_atom_index]} (in the original structure)."
                + " " + common_instruction
            )
        if self.mode_flag == "relative_to_motif_centroid":
            return (
                f"add [{motif.describe(**motif_desc_kwargs)}] to the structure,"
                f" with its centroid shifted in {coord_word} by"
                f" {describe_arraylike(self.relative_shift, precision=precision)} relative to the"
                f" centroid of [{rel_motif.describe(**relative_motif_desc_kwargs)}]."
                + " " + common_instruction
            )
        else:
            raise NotImplementedError(f"Invalid mode_flag: {self.mode_flag}")

    @classmethod
    def get_random_one(
            cls,
            operated_atoms: Atoms,
            seed: Optional[int] = None,
    ) -> "AddMotifAction":
        """Get a random AddMotifAction instance for the given operated motif.

        Args:
            operated_atoms (Atoms): The Atoms object that the motif is added to.
            seed (int, optional): Random seed for reproducibility. Default is None.
        Returns:
            AddMotifAction: A random instance of AddMotifAction.
        """
        rng = np.random.default_rng(seed)
        # Randomly select an additive motif to add.
        added_motif_probabilities = {
            "site": 0.5,
            "cluster": 0.5
        }
        class_alias = rng.choice(
            list(added_motif_probabilities.keys()),
            p=list(added_motif_probabilities.values())
        )
        operated_motif_kwargs = {
            "additive_mode": True,  # Always use additive mode.
            "class_alias": class_alias,
            "atoms": operated_atoms,
            "seed": seed,
        }
        # Choose a random cluster size if cluster motif is selected.
        if class_alias == "cluster":
            cluster_size = rng.integers(2, 5)  # Random cluster size between 2 and 5.
            operated_motif_kwargs["cluster_size"] = cluster_size
        operated_motif = get_random_motif(**operated_motif_kwargs)

        # Randomly select a mode based on probabilities.
        mode_flag = cls.get_random_mode(seed)

        kwargs = {
            "operated_motif": operated_motif,
            "operated_atoms": operated_atoms,
        }

        # randomize whether to use fractional coordinates
        use_fractional = rng.choice([True, False])

        if mode_flag == "absolute":
            if use_fractional:
                at_position = rng.uniform(size=3)  # Fractional position.
            else:
                cell = operated_atoms.cell.complete()
                at_position = rng.uniform(size=3) @ cell  # Cartesian position.
            kwargs["at_position"] = at_position
            kwargs["position_fractional"] = use_fractional
        elif mode_flag == "relative_to_position":
            if use_fractional:
                relative_to_position = rng.uniform(size=3)  # Fractional position.
                relative_shift = rng.uniform(-0.2, 0.2, size=3)  # Fractional shift.
            else:
                cell = operated_atoms.cell.complete()
                relative_to_position = rng.uniform(size=3) @ cell  # Cartesian position.
                relative_shift = rng.uniform(-0.2, 0.2, size=3) @ cell  # Cartesian shift.
            kwargs["relative_to_position"] = relative_to_position
            kwargs["relative_shift"] = relative_shift
            kwargs["position_fractional"] = use_fractional
        elif mode_flag == "relative_to_motif_centroid":
            # Randomly choose one motif class type.
            class_alias = rng.choice(
                ["site", "cluster", "bond"]
            )
            relative_motif_kwargs = {
                # Never use additive mode.
                "class_alias": class_alias,
                "atoms": operated_atoms,
                "seed": seed,
            }
            if class_alias == "cluster":
                cluster_size = rng.integers(2, 5)  # Random cluster size between 2 and 5.
                relative_motif_kwargs["cluster_size"] = cluster_size

            if class_alias != "site":
                relative_motif_kwargs["max_cluster_radius"] = 4.0  # Limit cluster radius to 4.0 Å.
            relative_to_motif = get_random_motif(**relative_motif_kwargs)
            if use_fractional:
                relative_shift = rng.uniform(-0.2, 0.2, size=3)  # Fractional shift.
            else:
                cell = operated_atoms.cell.complete()
                relative_shift = rng.uniform(-0.2, 0.2, size=3) @ cell  # Cartesian shift.
            kwargs["relative_to_motif"] = relative_to_motif
            kwargs["relative_shift"] = relative_shift
            kwargs["relative_style"] = "centroid_distance"
            kwargs["position_fractional"] = use_fractional
        elif mode_flag == "relative_to_pair_motif":
            # Randomly choose one motif class type.
            class_alias = rng.choice(
                ["cluster", "bond"]
            )
            relative_motif_kwargs = {
                # Never use additive mode.
                "class_alias": class_alias,
                "atoms": operated_atoms,
                "seed": seed,
                "max_cluster_radius": 4.0,  # Limit cluster radius to 4.0 Å.
            }
            if class_alias == "cluster":
                relative_motif_kwargs["cluster_size"] = 2  # Pair cluster.

            relative_to_motif = get_random_motif(**relative_motif_kwargs)
            relative_shift = rng.uniform(0.5, 3.0)  # Distance in angstroms.
            relative_atom_index = int(rng.choice([0, 1]))
            kwargs["relative_to_motif"] = relative_to_motif
            kwargs["relative_shift"] = relative_shift
            kwargs["relative_style"] = "position_in_line"
            kwargs["relative_atom_index"] = relative_atom_index
        else:
            raise NotImplementedError(f"Invalid mode_flag: {mode_flag}")

        return cls(**kwargs)
