"""Cell-related actions (v2) following the actions_v2 API.

Implements SuperCellActionV2 with randomize/execute/apply_random and config-driven
hyperparameters.
"""
from __future__ import annotations
import random
from typing import Optional, Dict, Any, Tuple
from ase import Atoms
from atom_world.actions_v2 import BaseActionV2


class SuperCellActionV2(BaseActionV2):
    """Create a supercell by scaling the cell by integer factors.

    Config keys supported (all optional):
      - super_min: minimum scale factor for each axis (int, default 1)
      - super_max: maximum scale factor for each axis (int, default 3)
      - max_volume: maximum product n1*n2*n3 allowed (int, default 8)
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config)

    def randomize(self, atoms: Atoms, rng=None) -> Dict[str, Any]:
        cfg = self.config
        smin = int(cfg.get('super_min', 1))
        smax = int(cfg.get('super_max', 3))
        max_vol = int(cfg.get('max_volume', 8))
        # sample until constraints satisfied (avoid trivial 1,1,1)
        for _ in range(50):
            n1 = random.randint(smin, smax)
            n2 = random.randint(smin, smax)
            n3 = random.randint(smin, smax)
            if n1 * n2 * n3 <= max_vol and not (n1 == 1 and n2 == 1 and n3 == 1):
                params = {"supercell_size": (n1, n2, n3)}
                self._last_params = params
                return params
        # fallback: choose smallest non-trivial scaling
        params = {"supercell_size": (2, 1, 1)}
        self._last_params = params
        return params

    def execute(self, atoms: Atoms, /, **params) -> Atoms:
        sc = params.get('supercell_size') or self._last_params.get('supercell_size')
        if sc is None:
            raise ValueError('supercell_size not provided')
        return atoms * tuple(sc)

    def __str__(self):
        sc = None
        if self._last_params:
            sc = self._last_params.get('supercell_size')
        return f"Create a supercell with size {sc if sc is not None else '(unknown)'}"
