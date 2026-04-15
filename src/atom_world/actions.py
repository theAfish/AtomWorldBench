# The basic actions to be performed in the cif files. 
# Such as adding atoms, removing atoms, and modifying atoms.
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from abc import ABC, abstractmethod


# Default hyperparameters for random sampling
DEFAULT_CONFIG = {
    "radius_min": 1.0,
    "radius_max": 4.0,
    "distance_min": 0.1,
    "distance_max": 3.0,
    "dpos_scale": 2.0,
    "distance_ratio_min": 0.1,
    "distance_ratio_max": 0.9,
    "angle_min": 45.0,
    "angle_max": 315.0,
    "symbol_pool": list(chemical_symbols[1:]),
    "decimal_places": 3,
}


def _get_config(config=None):
    """Merge user config with defaults."""
    if config is None:
        return dict(DEFAULT_CONFIG)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config)
    return cfg


def _safe_radius(atoms, index, rmin, rmax):
    """Adjust radius bounds to avoid deleting all atoms."""
    if index is not None and len(atoms) > 1:
        distances = atoms.get_distances(index, range(len(atoms)), mic=True)
        nonzero = distances[distances > 0]
        if len(nonzero) > 0:
            max_dist = np.max(nonzero)
            min_dist = np.min(nonzero)
            rmax = min(rmax, max_dist * 0.99)
            rmin = max(rmin, min_dist * 1.01)
            if rmax < rmin:
                rmax = max_dist * 0.5
            if rmax < 0.1:
                rmax = 0.1
    return rmin, rmax


class BaseAction(ABC):
    """Abstract base class for atom actions.

    Each subclass must implement ``execute`` and ``randomize``.
    ``randomize`` returns a dict of kwargs for ``__init__``,
    and ``apply_random`` is a convenience that randomizes + executes.
    """

    def __init__(self, atoms: Atoms):
        self.atoms = atoms

    def change_atoms(self, atoms: Atoms):
        """Change the atoms object for this action."""
        self.atoms = atoms

    @abstractmethod
    def execute(self):
        """Execute the action and return the modified Atoms object."""
        pass

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        """Sample random parameters for this action.

        Returns:
            dict of kwargs for ``__init__``, or None if no valid params found.
        """
        raise NotImplementedError(f"{cls.__name__} must implement randomize()")

    @classmethod
    def apply_random(cls, atoms, rng=None, config=None, copy=True):
        """Create a random action instance and execute it.

        Returns:
            (action, result_atoms) or (None, None) if no valid params found.
        """
        rng = rng if rng is not None else np.random.default_rng()
        target = atoms.copy() if copy else atoms
        params = cls.randomize(target, rng=rng, config=config)
        if params is None:
            return None, None
        action = cls(**params)
        result = action.execute()
        return action, result

    def __str__(self):
        return f"{self.__class__.__name__} action on {len(self.atoms)} atoms."

# Single atom actions
class AddAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, symbol: str, position: np.ndarray):
        super().__init__(atoms)
        self.symbol = symbol
        self.position = position

    def execute(self):
        new_atom = Atoms(symbols=self.symbol, positions=[self.position])
        self.atoms += new_atom
        return self.atoms

    def __str__(self):
        return f"Add one {self.symbol} atom at the Cartesian coordinate {self.position} to the cif file."

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        symbol = str(rng.choice(cfg['symbol_pool']))
        frac = rng.random(3)
        position = np.dot(frac, atoms.get_cell())
        position = np.round(position, decimals=cfg['decimal_places'])
        return {"atoms": atoms, "symbol": symbol, "position": position}


class RemoveAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, index: int):
        super().__init__(atoms)
        self.index = index
        
    def execute(self):
        if 0 <= self.index < len(self.atoms):
            del self.atoms[self.index]
            return self.atoms
        else:
            raise IndexError("Index out of bounds for atom removal.")
        
    def __str__(self):
        return f"Remove the atom at index {self.index} from the cif file. The indices of atoms are started from 0."

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) == 0:
            raise ValueError("No atoms to remove")
        rng = rng if rng is not None else np.random.default_rng()
        index = int(rng.integers(0, len(atoms)))
        return {"atoms": atoms, "index": index}


class MoveAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, index: int, d_pos: np.ndarray):
        super().__init__(atoms)
        self.index = index
        self.d_pos = d_pos

    def execute(self):
        if 0 <= self.index < len(self.atoms):
            self.atoms[self.index].position += self.d_pos
            return self.atoms
        else:
            raise IndexError("Index out of bounds for atom movement.")
        
    def __str__(self):
        return f"Move the atom at index {self.index} by {self.d_pos} angstrom in the cif file."

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) == 0:
            raise ValueError("No atoms to move")
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        index = int(rng.integers(0, len(atoms)))
        d_pos = rng.normal(scale=cfg['dpos_scale'], size=3)
        d_pos = np.round(d_pos, decimals=cfg['decimal_places'])
        return {"atoms": atoms, "index": index, "d_pos": d_pos}


class ChangeAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, index: int, symbol: str):
        super().__init__(atoms)
        self.index = index
        self.new_symbol = symbol

    def execute(self):
        if 0 <= self.index < len(self.atoms):
            self.atoms[self.index].symbol = self.new_symbol
            return self.atoms
        else:
            raise IndexError("Index out of bounds for atom modification.")
        
    def __str__(self):
        return f"Change the atom at index {self.index} into {self.new_symbol} in the cif file. The indices of atoms are started from 0."

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) == 0:
            raise ValueError("No atoms to change")
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        index = int(rng.integers(0, len(atoms)))
        current_symbol = atoms[index].symbol
        pool = [s for s in cfg['symbol_pool'] if s != current_symbol]
        symbol = str(rng.choice(pool))
        return {"atoms": atoms, "index": index, "symbol": symbol}


# double atom actions
class SwapAtomsAction(BaseAction):
    def __init__(self, atoms: Atoms, index1: int, index2: int):
        super().__init__(atoms)
        self.index1 = index1
        self.index2 = index2

    def execute(self):
        if (0 <= self.index1 < len(self.atoms)) and (0 <= self.index2 < len(self.atoms)):
            # Swap the types of the two atoms
            self.atoms[self.index1].symbol, self.atoms[self.index2].symbol = \
                self.atoms[self.index2].symbol, self.atoms[self.index1].symbol
            return self.atoms
        else:
            raise IndexError("Index out of bounds for atom swapping.")
        
    def __str__(self):
        # return f"Swap atoms at indices {self.index1} and {self.index2} in the cif file. The indices of atoms are started from 0."
        return f"Swap the spatial positions of atoms at indices {self.index1} and {self.index2} in the cif file. The indices of atoms are started from 0."

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) < 2:
            raise ValueError("Not enough atoms to swap")
        rng = rng if rng is not None else np.random.default_rng()
        for _ in range(20):
            idx1, idx2 = rng.choice(len(atoms), size=2, replace=False)
            if atoms[int(idx1)].symbol != atoms[int(idx2)].symbol:
                return {"atoms": atoms, "index1": int(idx1), "index2": int(idx2)}
        raise ValueError("Could not find two atoms of different types to swap.")


class InsertBetweenAtomsAction(BaseAction):
    def __init__(self, atoms: Atoms, index1: int, index2: int, symbol: str, distance_ratio: float):
        super().__init__(atoms)
        self.index1 = index1
        self.index2 = index2
        self.symbol = symbol
        self.distance_ratio = distance_ratio
        self.distance = 0
        # wrap indices to valid range
        self.index1 = self.index1 % len(self.atoms)
        self.index2 = self.index2 % len(self.atoms)

    def execute(self):
        if (0 <= self.index1 < len(self.atoms)) and (0 <= self.index2 < len(self.atoms)):
            relative_position = self.atoms[self.index2].position - self.atoms[self.index1].position
            self.distance = np.linalg.norm(relative_position) * self.distance_ratio
            position = self.atoms[self.index1].position + relative_position * self.distance_ratio
            new_atom = Atoms(symbols=self.symbol, positions=[position])
            self.atoms += new_atom
            self.atoms.set_pbc(self.atoms.get_pbc())  # Ensure periodic boundary
            return self.atoms
        else:
            raise IndexError("Index out of bounds for inserting atom between two atoms.")

    def __str__(self):
        return f"Insert a {self.symbol} atom in the line between atoms at indices {self.index1} and {self.index2}, and the inserted atom must be {self.distance:.2f} angstrom from atom at {self.index1} in the cif file."

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) < 2:
            raise ValueError("Not enough atoms to insert between")
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        idx1, idx2 = rng.choice(len(atoms), size=2, replace=False)
        symbol = str(rng.choice(cfg['symbol_pool']))
        dr = float(rng.uniform(cfg['distance_ratio_min'], cfg['distance_ratio_max']))
        dr = round(dr, cfg['decimal_places'])
        return {"atoms": atoms, "index1": int(idx1), "index2": int(idx2),
                "symbol": symbol, "distance_ratio": dr}


class MoveTowardsAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, index1: int, index2: int, distance: float):
        super().__init__(atoms)
        self.index1 = index1
        self.index2 = index2
        self.distance = distance

    def execute(self):
        if (0 <= self.index1 < len(self.atoms)) and (0 <= self.index2 < len(self.atoms)):
            direction = self.atoms[self.index2].position - self.atoms[self.index1].position
            norm = np.linalg.norm(direction)
            if norm == 0:
                raise ValueError("Cannot move towards an atom at the same position.")
            direction /= norm  # Normalize the direction vector
            move_vector = direction * self.distance
            self.atoms[self.index1].position += move_vector
            return self.atoms
        else:
            raise IndexError("Index out of bounds for moving towards another atom.")

    def __str__(self):
        return f"Move the atom at index {self.index1} towards the atom at index {self.index2} by {self.distance} angstrom in the cif file."

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) < 2:
            raise ValueError("Not enough atoms for MoveTowardsAtomAction")
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        idx1, idx2 = rng.choice(len(atoms), size=2, replace=False)
        distance = float(rng.uniform(cfg['distance_min'], cfg['distance_max']))
        distance = round(distance, cfg['decimal_places'])
        return {"atoms": atoms, "index1": int(idx1), "index2": int(idx2),
                "distance": distance}


# Multiple atom actions
class DeleteBelowAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, index: int, include_self: bool = False):
        super().__init__(atoms)
        self.index = index
        self.include_self = include_self

    def execute(self):
        z_threshold = self.atoms[self.index].position[2]
        indices_to_delete = [i for i, atom in enumerate(self.atoms) if atom.position[2] < z_threshold]
        if not self.include_self:
            indices_to_delete = [i for i in indices_to_delete if i != self.index]
        if indices_to_delete:
            self.atoms = self.atoms[[i for i in range(len(self.atoms)) if i not in indices_to_delete]]
            return self.atoms
        else:
            raise ValueError("No atoms below the specified atom to delete.")
        
    def __str__(self):
        # return f"Delete all atoms below the atom at index {self.index} in the cif file." + (" Including itself" if self.include_self else " Excluding itself") + " and atoms with the same z coordinate."
        return f"Delete all atoms whose z coordinate is lower than the atom at index {self.index} in the cif file." + (" Including itself" if self.include_self else " Excluding itself") + " and atoms with the same z coordinate."

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) < 2:
            return None
        rng = rng if rng is not None else np.random.default_rng()
        for _ in range(20):
            idx = int(rng.integers(0, len(atoms)))
            z_threshold = atoms[idx].position[2]
            indices_below = [i for i, a in enumerate(atoms)
                             if a.position[2] < z_threshold and i != idx]
            if indices_below:
                include_self = bool(rng.random() < 0.5)
                return {"atoms": atoms, "index": idx, "include_self": include_self}
        return None


class DeleteAroundAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, index: int, radius: float):
        super().__init__(atoms)
        self.index = index
        self.radius = radius

    def execute(self):
        if 0 <= self.index < len(self.atoms):
            distances = self.atoms.get_distances(self.index, range(len(self.atoms)), mic=True)
            indices_to_delete = np.where(distances < self.radius)[0]
            if len(indices_to_delete) > 0:
                self.atoms = self.atoms[[i for i in range(len(self.atoms)) if i not in indices_to_delete]]
                return self.atoms
            else:
                raise ValueError("No atoms found within the specified radius to delete.")
        else:
            raise IndexError("Index out of bounds for deleting around an atom.")

    def __str__(self):
        return f"Delete all atoms within {self.radius} angstrom around the atom at index {self.index} in the cif file."

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) == 0:
            raise ValueError("No atoms")
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        index = int(rng.integers(0, len(atoms)))
        rmin, rmax = _safe_radius(atoms, index, cfg['radius_min'], cfg['radius_max'])
        radius = float(rng.uniform(rmin, rmax))
        radius = round(radius, cfg['decimal_places'])
        return {"atoms": atoms, "index": index, "radius": radius}


class MoveSelectedAtomsAction(BaseAction):
    def __init__(self, atoms: Atoms, indices: list[int], d_pos: np.ndarray):
        super().__init__(atoms)
        self.indices = indices
        self.d_pos = d_pos

    def execute(self):
        for index in self.indices:
            if 0 <= index < len(self.atoms):
                self.atoms[index].position += self.d_pos
            else:
                raise IndexError(f"Index {index} out of bounds for atom movement.")
        return self.atoms
    
    def __str__(self):
        return f"Move atoms at indices {self.indices} by {self.d_pos} angstrom in the cif file."

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) == 0:
            raise ValueError("No atoms")
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        count = int(rng.integers(1, len(atoms) + 1))
        indices = list(rng.choice(len(atoms), size=count, replace=False))
        d_pos = rng.normal(scale=cfg['dpos_scale'], size=3)
        d_pos = np.round(d_pos, decimals=cfg['decimal_places'])
        return {"atoms": atoms, "indices": [int(i) for i in indices], "d_pos": d_pos}


class MoveAroundAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, index: int, radius: float, d_pos: np.ndarray):
        super().__init__(atoms)
        self.index = index
        self.radius = radius
        self.d_pos = d_pos

    def execute(self):
        if 0 <= self.index < len(self.atoms):
            # center_position = self.atoms[self.index].position
            # distances = np.linalg.norm(self.atoms.positions - center_position, axis=1)
            distances = self.atoms.get_distances(self.index, range(len(self.atoms)), mic=True)
            indices_to_move = np.where(distances < self.radius)[0]
            for i in indices_to_move:
                self.atoms[i].position += self.d_pos
            return self.atoms
        else:
            raise IndexError("Index out of bounds for moving around an atom.")

    def __str__(self):
        return f"Move all surrounding atoms within {self.radius} angstrom around the center atom at index {self.index} by {self.d_pos} angstrom in the cif file."

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) == 0:
            raise ValueError("No atoms")
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        index = int(rng.integers(0, len(atoms)))
        rmin, rmax = _safe_radius(atoms, index, cfg['radius_min'], cfg['radius_max'])
        radius = float(rng.uniform(rmin, rmax))
        radius = round(radius, cfg['decimal_places'])
        d_pos = rng.normal(scale=cfg['dpos_scale'], size=3)
        d_pos = np.round(d_pos, decimals=cfg['decimal_places'])
        return {"atoms": atoms, "index": index, "radius": radius, "d_pos": d_pos}


class RotateAroundAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, index: int, radius: float, angle: float, axis: np.ndarray):
        super().__init__(atoms)
        self.index = index
        self.radius = radius
        if not (0 <= angle < 360):
            raise ValueError("Angle must be in the range [0, 2π).")
        if np.linalg.norm(axis) == 0:
            raise ValueError("Axis of rotation cannot be a zero vector.")
        self.angle = angle
        self.axis = axis / np.linalg.norm(axis)  # Normalize the rotation axis

    def execute(self):
        if 0 <= self.index < len(self.atoms):
            center_position = self.atoms[self.index].position
            distances = self.atoms.get_distances(self.index, range(len(self.atoms)), mic=True)
            indices_to_rotate = np.where(distances < self.radius)[0]
            indices_to_rotate = [i for i in indices_to_rotate if i != self.index]
            if not indices_to_rotate:
                return self.atoms  # Nothing to rotate
            # Extract the atoms to rotate
            sub_atoms = self.atoms[indices_to_rotate]
            # Rotate in-place around the axis and center
            sub_atoms.rotate(self.angle, self.axis, center=center_position)
            # Update positions in the main atoms object
            for idx, sub_atom in zip(indices_to_rotate, sub_atoms):
                self.atoms[idx].position = sub_atom.position
            return self.atoms
        else:
            raise IndexError("Index out of bounds for rotating around an atom.")

    def __str__(self):
        return f"Rotate all surrounding atoms within {self.radius} angstrom of the center atom at index {self.index} by {self.angle} degree around the axis {self.axis} in the cif file. The rotation should following the right-hand rule."

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) == 0:
            raise ValueError("No atoms")
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        index = int(rng.integers(0, len(atoms)))
        rmin, rmax = _safe_radius(atoms, index, cfg['radius_min'], cfg['radius_max'])
        radius = float(rng.uniform(rmin, rmax))
        radius = round(radius, cfg['decimal_places'])
        angle = float(rng.uniform(cfg['angle_min'], cfg['angle_max']))
        angle = round(angle, cfg['decimal_places'])
        axes = np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]])
        axis = axes[int(rng.integers(0, len(axes)))]
        return {"atoms": atoms, "index": index, "radius": radius,
                "angle": angle, "axis": axis}


class RotateWholeAction(BaseAction):
    def __init__(self, atoms: Atoms, angle: float, axis: np.ndarray):
        super().__init__(atoms)
        if not (0 <= angle < 360):
            raise ValueError("Angle must be in the range [0, 2π).")
        if np.linalg.norm(axis) == 0:
            raise ValueError("Axis of rotation cannot be a zero vector.")
        self.angle = angle
        self.axis = axis / np.linalg.norm(axis)  # Normalize the rotation axis

    def execute(self):
        self.atoms.rotate(self.angle, self.axis, rotate_cell=True)
        return self.atoms

    def __str__(self):
        return f"Rotate the structure and cell by {self.angle} degree around the axis {self.axis} in the cif file. The rotation should following the right-hand rule."

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        angle = float(rng.uniform(cfg['angle_min'], cfg['angle_max']))
        angle = round(angle, cfg['decimal_places'])
        axes = np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]])
        axis = axes[int(rng.integers(0, len(axes)))]
        return {"atoms": atoms, "angle": angle, "axis": axis}

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        angle = float(rng.uniform(cfg['angle_min'], cfg['angle_max']))
        angle = round(angle, cfg['decimal_places'])
        axes = np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]])
        axis = axes[int(rng.integers(0, len(axes)))]
        return {"atoms": atoms, "angle": angle, "axis": axis}
    

class MoveAllAction(BaseAction):
    def __init__(self, atoms: Atoms, d_pos: np.ndarray):
        super().__init__(atoms)
        self.d_pos = d_pos

    def execute(self):
        unwrapped_positions = self.atoms.get_positions(wrap=False)
        self.atoms.set_positions(unwrapped_positions + self.d_pos)
        return self.atoms

    def __str__(self):
        return f"Move all the atoms in the structure by {self.d_pos} angstrom in the cif file. Do not need to consider periodic boundary conditions. Please keep the cell and the order of atoms unchanged."

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        d_pos = rng.normal(scale=cfg['dpos_scale'], size=3)
        d_pos = np.round(d_pos, decimals=cfg['decimal_places'])
        return {"atoms": atoms, "d_pos": d_pos}
    

if __name__ == "__main__":
    atoms = Atoms("H2O", positions=[[0.1, 0, 0], [0.1, 0, 1], [1.1, 0, 0]], cell=[2, 2, 2], pbc=True)
    atoms.write("original_water.cif", wrap=False)
    mv_all = MoveAllAction(atoms, np.array([-0.2, 0., 0.]))
    mv_all.execute()
    print(atoms.positions)
    # save to cif
    atoms.write("moved_water.cif", wrap=False)

    # load the cif file
    from pymatgen.io.cif import CifParser
    parser = CifParser("moved_water.cif")
    structures = parser.parse_structures(primitive=False)
    print(structures[0])