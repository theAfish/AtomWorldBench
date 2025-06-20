# The basic actions to be performed in the cif files. 
# Such as adding atoms, removing atoms, and modifying atoms.
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols

class BaseAction:
    def __init__(self, atoms: Atoms):
        self.atoms = atoms

    def change_atoms(self, atoms: Atoms):
        """Change the atoms object for this action."""
        self.atoms = atoms

    def execute(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

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
        return f"Add one {self.symbol} at {self.position} to the cif file."
    
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
        return f"Remove the atom at index {self.index} from the cif file."
    
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
        return f"Move the atom at index {self.index} by {self.d_pos} in the cif file."
    
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
        return f"Change the atom at index {self.index} into {self.new_symbol} in the cif file."

# double atom actions
class SwapAtomsAction(BaseAction):
    def __init__(self, atoms: Atoms, index1: int, index2: int):
        super().__init__(atoms)
        self.index1 = index1
        self.index2 = index2

    def execute(self):
        if (0 <= self.index1 < len(self.atoms)) and (0 <= self.index2 < len(self.atoms)):
            self.atoms[self.index1], self.atoms[self.index2] = self.atoms[self.index2], self.atoms[self.index1]
            return self.atoms
        else:
            raise IndexError("Index out of bounds for atom swapping.")
        
    def __str__(self):
        return f"Swap atoms at indices {self.index1} and {self.index2} in the cif file."
    
class InsertBetweenAtomsAction(BaseAction):
    def __init__(self, atoms: Atoms, index1: int, index2: int, symbol: str, distance_ratio: float):
        super().__init__(atoms)
        self.index1 = index1
        self.index2 = index2
        self.symbol = symbol
        self.distance_ratio = distance_ratio

    def execute(self):
        if (0 <= self.index1 < len(self.atoms)) and (0 <= self.index2 < len(self.atoms)):
            relative_position = self.atoms[self.index2].position - self.atoms[self.index1].position
            position = self.atoms[self.index1].position + relative_position * self.distance_ratio
            new_atom = Atoms(symbols=self.symbol, positions=[position])
            self.atoms = self.atoms[:self.index1 + 1] + new_atom + self.atoms[self.index2:]
            self.atoms.set_pbc(self.atoms.get_pbc())  # Ensure periodic boundary
            return self.atoms
        else:
            raise IndexError("Index out of bounds for inserting atom between two atoms.")

    def __str__(self):
        return f"Insert {self.symbol} between atoms at indices {self.index1} and {self.index2} that are {self.distance_ratio:.2f} of the way from {self.index1} to {self.index2} in the cif file."
    
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
        return f"Move atom at index {self.index1} towards atom at index {self.index2} by {self.distance} in the cif file."
    
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
        return f"Delete all atoms below the atom at index {self.index} in the cif file." + (" Including itself." if self.include_self else " Excluding itself.")
    
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
        return f"Delete all atoms within {self.radius} of the atom at index {self.index} in the cif file."

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
        return f"Move atoms at indices {self.indices} by {self.d_pos} in the cif file."
    
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
        return f"Move all surrounding atoms within {self.radius} of the center atom at index {self.index} by {self.d_pos} in the cif file."
    
class RotateAroundAtomAction(BaseAction):
    def __init__(self, atoms: Atoms, index: int, radius: float, angle: float, axis: np.ndarray):
        super().__init__(atoms)
        self.index = index
        self.radius = radius
        if not (0 <= angle < 2 * np.pi):
            raise ValueError("Angle must be in the range [0, 2π).")
        if np.linalg.norm(axis) == 0:
            raise ValueError("Axis of rotation cannot be a zero vector.")
        self.angle = angle
        self.axis = axis / np.linalg.norm(axis)  # Normalize the rotation axis

    def execute(self):
        if 0 <= self.index < len(self.atoms):
            from ase.geometry import rotate
            center_position = self.atoms[self.index].position
            distances = self.atoms.get_distances(self.index, range(len(self.atoms)), mic=True)
            indices_to_rotate = np.where(distances < self.radius)[0]
            for i in indices_to_rotate:
                if i != self.index:
                    # Rotate the atom around the specified axis
                    self.atoms[i].position = rotate(self.atoms[i].position - center_position, self.angle, self.axis) + center_position
            return self.atoms
        else:
            raise IndexError("Index out of bounds for rotating around an atom.")

    def __str__(self):
        return f"Rotate  all surrounding atoms within {self.radius} of the center atom at index {self.index} by {self.angle} radians around the axis {self.axis} in the cif file."