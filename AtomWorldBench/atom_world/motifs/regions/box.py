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
            boundary_fractional: bool = True,
            tol: float = 1e-6,
    ):
        """Initialize the box region motif.

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
            boundary_fractional (bool): Whether the provided boundaries are in fractional coordinates.
                If False, the boundaries are in cartesian coordinates.
            tol (float): Tolerance for boundary inclusion. Default is 1e-6.
        """
        BaseRegionMotif.__init__(self, in_atoms)
        self.boundary_fractional = boundary_fractional
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.tol = tol

    def _get_site_indices_offsets_in_atoms(self) -> tuple[list[int], ndarray]:
        """Get the indices and periodic offsets of sites in this box region within the parent atoms.

        Returns:
            indices (list[int]): A list of indices of sites in this box region within the parent atoms.
            offsets (list[ndarray]): A list of periodic offsets for each site in this box region relative
                to the parent atoms.get_positions(wrap=False).
                Each offset is a numpy array of shape (3,).
        """
        if self.boundary_fractional:
            frac_coords = self.in_atoms.get_scaled_positions(wrap=True)
            xmin = self.xmin
            xmax = self.xmax
            ymin = self.ymin
            ymax = self.ymax
            zmin = self.zmin
            zmax = self.zmax
        else:
            cart_coords = self.in_atoms.get_positions(wrap=True)
            cell = self.in_atoms.cell.complete()
            frac_coords = self.in_atoms.get_scaled_positions(wrap=True)
            xmin = None if self.xmin is None\
                else np.linalg.solve(cell.T, np.array([self.xmin, 0, 0])).item()
            xmax = None if self.xmax is None\
                else np.linalg.solve(cell.T, np.array([self.xmax, 0, 0])).item()
            ymin = None if self.ymin is None\
                else np.linalg.solve(cell.T, np.array([0, self.ymin, 0])).item()
            ymax = None if self.ymax is None\
                else np.linalg.solve(cell.T, np.array([0, self.ymax, 0])).item()
            zmin = None if self.zmin is None\
                else np.linalg.solve(cell.T, np.array([0, 0, self.zmin])).item()
            zmax = None if self.zmax is None\
                else np.linalg.solve(cell.T, np.array([0, 0, self.zmax])).item()

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

        filt = (xmin - self.tol <= frac_coords[:, 0] <= xmax + self.tol) & \
               (ymin - self.tol <= frac_coords[:, 1] <= ymax + self.tol) & \
               (zmin - self.tol <= frac_coords[:, 2] <= zmax + self.tol)
        indices = np.where(filt)[0].tolist()
        offsets = np.zeros((len(indices), 3), dtype=int)
        return indices, offsets

    def _get_default_name(self) -> str:
        return self.__class__.__name__

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
        def _get_directional_condition(d="x"):
            min = getattr(self, f"{d}min")
            max = getattr(self, f"{d}max")
            if (min is not None) and (max is not None):
                return f"{min:{precision}.f} ≤ {d} ≤ {max:{precision}.f}"
            elif min is not None:
                return f"{d} ≥ {min:{precision}.f}"
            elif max is not None:
                return f"{d} ≤ {max:{precision}.f}"
            else:
                return None

        x_condition = _get_directional_condition("x")
        y_condition = _get_directional_condition("y")
        z_condition = _get_directional_condition("z")
        condition_str = ", ".join(
            cond for cond in [x_condition, y_condition, z_condition] if cond is not None
        )
        boundary_type = "fractional" if self.boundary_fractional else "cartesian"
        description = f"all atoms with {boundary_type} coordinates satisfying: {condition_str}."
        return description

    @classmethod
    def detect_random_one(
            cls,
            atoms: Atoms,
            seed: Optional[int] = None,
            **kwargs,
    ) -> "BoxRegionMotif":
        """Detect a random box region motif within the given atoms.

        Args:
            atoms (Atoms): The atoms to detect the box region motif from.
            seed (Optional[int]): Random seed for reproducibility. Default is None.
            **kwargs: Additional keyword arguments for BoxRegionMotif initialization.

        Returns:
            BoxRegionMotif: A randomly detected box region motif.
        """
        atoms.wrap()
        rng = np.random.default_rng(seed)
        # Randomly determine whether each direction has min and/or max
        def random_boundary():
            has_min = rng.choice([True, False])
            has_max = rng.choice([True, False])
            return has_min, has_max
        x_has_min, x_has_max = random_boundary()
        y_has_min, y_has_max = random_boundary()
        z_has_min, z_has_max = random_boundary()
        # Randomly select min and max for each dimension
        x_min = rng.uniform(0, 1) if x_has_min else None
        x_max = rng.uniform(x_min or 0, 1) if x_has_max else None
        y_min = rng.uniform(0, 1) if y_has_min else None
        y_max = rng.uniform(y_min or 0, 1) if y_has_max else None
        z_min = rng.uniform(0, 1) if z_has_min else None
        z_max = rng.uniform(z_min or 0, 1) if z_has_max else None
        return cls(
            in_atoms=atoms,
            xmin=x_min,
            xmax=x_max,
            ymin=y_min,
            ymax=y_max,
            zmin=z_min,
            zmax=z_max,
            boundary_fractional=True,  # Always use fractional boundary.
        )
