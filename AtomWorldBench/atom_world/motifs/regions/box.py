"""Select all atoms within a specified fractional or cartesian coordinates range.

Can be used to implement operations such as slab creation.
Does not allow selecting arbitrary rotated boxes, only axis-aligned boxes.
"""

from typing import Optional

from ase import Atoms
import numpy as np
from numpy import ndarray

from .base import BaseRegionMotif
from ..base import BaseMotif
from ....common.globals import DEFAULT_FLOAT_TO_STRING_PRECISION

from ....common.registry import register


@register(BaseMotif, ["box", "box-region"])
@register(BaseRegionMotif, ["box", "box-region"])
class BoxRegionMotif(BaseRegionMotif):
    """A box region motif that selects all atoms within a specified fractional or cartesian coordinates range.

    Can be used to implement operations such as slab creation.
    """
    def __init__(
            self,
            in_atoms: Atoms,
            xmin: Optional[float] = None,
            xmax: Optional[float] = None,
            ymin: Optional[float] = None,
            ymax: Optional[float] = None,
            zmin: Optional[float] = None,
            zmax: Optional[float] = None,
            tol: float = 1e-6,
            symbols: Optional[list[str]] = None,
    ):
        """Initialize the box region motif.

        All boundaries must be provided in fractional coordinates.
        Args:
            in_atoms (Atoms): The atoms that this region motif is in.
                Notice: this object will always be wrapped at init if not already!
                All cell offsets will be computed relative to the wrapped positions.
            xmin (float, optional): The minimum x coordinate of the box.
                If None, no minimum x boundary is applied.
            xmax (float, optional): The maximum x coordinate of the box.
                If None, no maximum x boundary is applied.
            ymin (float, optional): The minimum y coordinate of the box.
                If None, no minimum y boundary is applied.
            ymax (float, optional): The maximum y coordinate of the box.
                If None, no maximum y boundary is applied.
            zmin (float, optional): The minimum z coordinate of the box.
                If None, no minimum z boundary is applied.
            zmax (float, optional): The maximum z coordinate of the box.
                If None, no maximum z boundary is applied.
            tol (float): Tolerance for boundary inclusion. Default is 1e-6.
            symbols (list[str], optional): A list of chemical symbols that this motif includes.
                Other elements will not be selected as part of this motif.
        """
        BaseRegionMotif.__init__(self, in_atoms)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.tol = tol
        self.symbols = symbols

    def _get_site_indices_offsets_in_atoms(self) -> tuple[list[int], ndarray]:
        """Get the indices and periodic offsets of sites in this box region within the parent atoms.

        Returns:
            indices (list[int]): A list of indices of sites in this box region within the parent atoms.
            offsets (list[ndarray]): A list of periodic offsets for each site in this box region relative
                to the parent atoms.get_positions(wrap=False).
                Each offset is a numpy array of shape (3,).
        """
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        zmin = self.zmin
        zmax = self.zmax
        if xmin is None:
            xmin = -np.inf
        if xmax is None:
            xmax = np.inf
        if ymin is None:
            ymin = -np.inf
        if ymax is None:
            ymax = np.inf
        if zmin is None:
            zmin = -np.inf
        if zmax is None:
            zmax = np.inf

        filt_coords = self.in_atoms.get_scaled_positions(wrap=True)

        filt = (
                (xmin - self.tol <= filt_coords[:, 0]) &
                (filt_coords[:, 0] <= xmax + self.tol) &
                (ymin - self.tol <= filt_coords[:, 1]) &
                (filt_coords[:, 1] <= ymax + self.tol) &
                (zmin - self.tol <= filt_coords[:, 2]) &
                (filt_coords[:, 2] <= zmax + self.tol)
        )

        if self.symbols is not None:
            filt = filt & np.isin(self.in_atoms.get_chemical_symbols(), self.symbols)
        indices = np.where(filt)[0].tolist()
        offsets = np.zeros((len(indices), 3), dtype=int)
        return indices, offsets

    def _get_default_name(self) -> str:
        return "a box region"

    def describe(
            self,
            precision: int = DEFAULT_FLOAT_TO_STRING_PRECISION,
    ) -> str:
        """Describe the box region motif.

        Args:
            precision (int): The precision for floating point numbers in the description.
                Default is DEFAULT_FLOAT_TO_STRING_PRECISION.

        Returns:
            str: A description of the box region motif.
        """
        def _get_directional_condition(prec, d="x"):
            rmin = getattr(self, f"{d}min")
            rmax = getattr(self, f"{d}max")
            if (rmin is not None) and (rmax is not None):
                return f"{rmin:.{prec}f} ≤ {d} ≤ {rmax:.{prec}f}"
            elif rmin is not None:
                return f"{d} ≥ {rmin:.{prec}f}"
            elif rmax is not None:
                return f"{d} ≤ {rmax:.{prec}f}"
            else:
                return None

        x_condition = _get_directional_condition(precision, "x")
        y_condition = _get_directional_condition(precision, "y")
        z_condition = _get_directional_condition(precision, "z")
        condition_str = ", ".join(
            cond for cond in [x_condition, y_condition, z_condition] if cond is not None
        )
        symbol_word = (
            "all atoms" if self.symbols is None
            else f"all atoms with element symbols {self.symbols}"
        )
        return f"{symbol_word} with fractional coordinates satisfying: {condition_str}."

    @classmethod
    def detect_random_one(
            cls,
            atoms: Atoms,
            randomize_symbols: bool = False,
            randomize_boundaries: bool = False,
            seed: Optional[int] = None,
    ) -> "BoxRegionMotif":
        """Detect a random box region motif within the given atoms.

        Args:
            atoms (Atoms): The atoms to detect the box region motif from.\
            randomize_boundaries (bool): If True, whether min and max exist for each direction
                will be randomly chosen. If False, all directions will have both min and max.
                Defaults to False.
            randomize_symbols (bool): If True, the symbols of the atoms in the motif will be
                randomly chosen from the symbols of the atoms in the provided Atoms object.
                If False, the symbols will be set to None, meaning all atoms in the region
                will be included regardless of their symbols. Defaults to False.
            seed (Optional[int]): Random seed for reproducibility. Default is None.

        Returns:
            BoxRegionMotif: A randomly detected box region motif.
        """
        atoms.wrap()
        if len(atoms) == 0:
            raise ValueError("Cannot detect a box region motif from an empty Atoms object.")
        rng = np.random.default_rng(seed)
        # Randomly determine whether each direction has min and/or max
        def random_boundary():
            has_min = rng.choice([True, False])
            has_max = rng.choice([True, False])
            return has_min, has_max
        if randomize_boundaries:
            x_has_min, x_has_max = random_boundary()
            y_has_min, y_has_max = random_boundary()
            z_has_min, z_has_max = random_boundary()
        else:
            x_has_min, x_has_max = True, True
            y_has_min, y_has_max = True, True
            z_has_min, z_has_max = True, True

        # Get a random box within the cell that guarantees at least one atom in it.
        scaled_positions = atoms.get_scaled_positions(wrap=True)
        dists_to_center = np.linalg.norm(
            scaled_positions - 0.5, axis=1
        )
        weights_to_center = np.exp(-5.0 * dists_to_center)
        # Choose a random point to build box around (this will guarantee non-empty box)
        center_index = rng.choice(len(atoms), p=weights_to_center / weights_to_center.sum())
        center = scaled_positions[center_index]
        # Define box limits.
        def get_limit(center_coord, has_min, has_max):
            min_upper = max(0, center_coord - 0.05)
            if min_upper == 0:
                min_lim = 0
            else:
                min_lim = rng.uniform(0.0, min_upper)
            max_lower = min(1.0, center_coord + 0.05)
            if max_lower == 1.0:
                max_lim = 1.0
            else:
                max_lim = rng.uniform(max_lower, 1.0)
            if has_min and has_max:
                return min_lim, max_lim
            elif has_min:
                return min_lim, None
            elif has_max:
                return None, max_lim
            else:
                return None, None

        x_min, x_max = get_limit(center[0], x_has_min, x_has_max)
        y_min, y_max = get_limit(center[1], y_has_min, y_has_max)
        z_min, z_max = get_limit(center[2], z_has_min, z_has_max)

        if randomize_symbols:
            unique_symbols = list(set(atoms.get_chemical_symbols()))
            n_symbols = rng.integers(1, len(unique_symbols) + 1)
            symbols = rng.choice(unique_symbols, size=n_symbols, replace=False).tolist()
        else:
            symbols = None

        return cls(
            in_atoms=atoms,
            xmin=x_min,
            xmax=x_max,
            ymin=y_min,
            ymax=y_max,
            zmin=z_min,
            zmax=z_max,
            symbols=symbols,
        )
