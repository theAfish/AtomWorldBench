"""Simple (index-based) rotate atom actions."""
import numpy as np
from ase import Atoms

from .base import BaseAction, _get_config, _safe_radius

_STANDARD_AXES = np.array([
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [-1, 0, 0], [0, -1, 0], [0, 0, -1],
])


class RotateAroundAtomAction(BaseAction):
    def __init__(
        self,
        atoms: Atoms,
        index: int,
        radius: float,
        angle: float,
        axis: np.ndarray,
    ):
        super().__init__(atoms)
        if not (0 <= angle < 360):
            raise ValueError("Angle must be in the range [0, 360).")
        if np.linalg.norm(axis) == 0:
            raise ValueError("Axis of rotation cannot be a zero vector.")
        self.index = index
        self.radius = radius
        self.angle = angle
        self.axis = axis / np.linalg.norm(axis)

    def execute(self):
        if 0 <= self.index < len(self.atoms):
            center_position = self.atoms[self.index].position
            distances = self.atoms.get_distances(
                self.index, range(len(self.atoms)), mic=True
            )
            indices_to_rotate = [
                i for i in np.where(distances < self.radius)[0] if i != self.index
            ]
            if not indices_to_rotate:
                return self.atoms
            sub_atoms = self.atoms[indices_to_rotate]
            sub_atoms.rotate(self.angle, self.axis, center=center_position)
            for idx, sub_atom in zip(indices_to_rotate, sub_atoms):
                self.atoms[idx].position = sub_atom.position
            return self.atoms
        raise IndexError("Index out of bounds for rotating around an atom.")

    def __str__(self):
        return (
            f"Rotate all surrounding atoms within {self.radius} angstrom of the"
            f" center atom at index {self.index} by {self.angle} degree around the"
            f" axis {self.axis} in the cif file. The rotation should following the"
            " right-hand rule."
        )

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) == 0:
            raise ValueError("No atoms")
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        index = int(rng.integers(0, len(atoms)))
        rmin, rmax = _safe_radius(atoms, index, cfg["radius_min"], cfg["radius_max"])
        radius = round(float(rng.uniform(rmin, rmax)), cfg["decimal_places"])
        angle = round(float(rng.uniform(cfg["angle_min"], cfg["angle_max"])), cfg["decimal_places"])
        axis = _STANDARD_AXES[int(rng.integers(0, len(_STANDARD_AXES)))]
        return {"atoms": atoms, "index": index, "radius": radius, "angle": angle, "axis": axis}


class RotateWholeAction(BaseAction):
    def __init__(self, atoms: Atoms, angle: float, axis: np.ndarray):
        super().__init__(atoms)
        if not (0 <= angle < 360):
            raise ValueError("Angle must be in the range [0, 360).")
        if np.linalg.norm(axis) == 0:
            raise ValueError("Axis of rotation cannot be a zero vector.")
        self.angle = angle
        self.axis = axis / np.linalg.norm(axis)

    def execute(self):
        self.atoms.rotate(self.angle, self.axis, rotate_cell=True)
        return self.atoms

    def __str__(self):
        return (
            f"Rotate the structure and cell by {self.angle} degree around the axis"
            f" {self.axis} in the cif file. The rotation should following the right-hand rule."
        )

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        angle = round(float(rng.uniform(cfg["angle_min"], cfg["angle_max"])), cfg["decimal_places"])
        axis = _STANDARD_AXES[int(rng.integers(0, len(_STANDARD_AXES)))]
        return {"atoms": atoms, "angle": angle, "axis": axis}
