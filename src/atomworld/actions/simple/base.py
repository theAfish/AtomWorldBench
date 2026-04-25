"""Base class and shared configuration for simple (index-based) atom actions."""
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
    """Abstract base class for simple (index-based) atom actions.

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
