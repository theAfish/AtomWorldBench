"""Dataset generator for the RemoveMoleculeAction active task.

Pipeline
--------
For each sample:

1. **Base structure** — load a bulk CIF, convert to the conventional standard
   cell via ``SpacegroupAnalyzer``.
2. **Surface construction** — cut a slab using ASE's ``surface()`` helper with
   a randomly sampled number of layers (``n_layers_range``), then make it
   3D-periodic.  The slab is converted to a pymatgen Structure for downstream
   use.
3. **Supercell** — expand the base slab so the surface area comfortably
   accommodates the chosen molecule.  A random base size (``supercell_range``)
   is chosen and then bumped up if the molecule footprint is too large.
4. **Adsorbate injection** — sample a molecule from the ZINC SMILES library,
   build 3D coordinates via RDKit ETKDGv3 + MMFF94, then place it on the slab
   via pymatgen's ``AdsorbateSiteFinder``.  Placement is validated for minimum
   separation.
5. **Target** — the clean slab (before adsorbate injection).
6. **Output** — yields dataset items ready for JSON serialisation.

Each item carries:
    ``action_type``   "RemoveMoleculeAction"
    ``task_category`` "active"
    ``problem_id``    UUID
    ``mp_id``         source CIF stem (e.g. "mp-1234")
    ``action_prompt`` "Remove the molecule(s) inside the structure."
    ``input``         CIF string of slab + adsorbate
    ``output``        CIF string of clean slab
    ``verifiers``     ["output_format", "cif_parsing", "atom_count",
                       "structure_match"]
    ``metadata``      molecule SMILES/formula, slab composition, supercell
                      params, miller index, etc.

Dependencies (datagen extras)
------------------------------
    ase ≥ 3.23, pymatgen (core dep), rdkit
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from ase.build import surface as ase_build_surface
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from rdkit import Chem
from rdkit.Chem import AllChem

from data_generation.base_data_generator import BaseDataGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path to the ZINC SMILES file (relative to project root).
# Resolved at import time based on this file's location:
#   src/data_generation/ → ../../data/molecules/zinc.txt
_DEFAULT_ZINC_PATH: str = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "data", "molecules", "zinc.txt")
)

DEFAULT_MILLER_INDICES: List[Tuple[int, int, int]] = [
    (0, 0, 1),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
]

VACUUM: float = 10.0        # Å vacuum above slab (passed to ASE surface())
MIN_SEPARATION: float = 1.8  # Å — minimum adsorbate-to-slab distance
ADSORPTION_HEIGHT: float = 2.0  # Å — height above topmost surface atom

# ---------------------------------------------------------------------------
# ZINC molecule library
# ---------------------------------------------------------------------------

_ZINC_SMILES_CACHE: Optional[List[str]] = None


def _load_zinc_smiles(path: str) -> List[str]:
    """Load and cache the ZINC SMILES list from *path* (one SMILES per line)."""
    global _ZINC_SMILES_CACHE
    if _ZINC_SMILES_CACHE is None:
        with open(path, "r", encoding="utf-8") as fh:
            _ZINC_SMILES_CACHE = [ln.strip() for ln in fh if ln.strip()]
        logger.info("Loaded %d ZINC SMILES from %s", len(_ZINC_SMILES_CACHE), path)
    return _ZINC_SMILES_CACHE


def _build_molecule_from_smiles(smiles: str) -> Optional[Molecule]:
    """Generate a 3D-embedded pymatgen Molecule from *smiles* via RDKit.

    Uses ETKDGv3 conformer generation and MMFF94 optimisation.
    Returns ``None`` on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        result = AllChem.EmbedMolecule(mol)
        if result != 0:
            return None

    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass

    conf = mol.GetConformer()
    species = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = [[*conf.GetAtomPosition(i)] for i in range(mol.GetNumAtoms())]
    return Molecule(species, coords)


# ---------------------------------------------------------------------------
# Bulk loading
# ---------------------------------------------------------------------------

def _load_bulk_structure(cif_path: str):
    """Load a bulk pymatgen Structure from a CIF file."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser = CifParser(cif_path)
            structures = parser.parse_structures(primitive=False, check_occu=False)
        return structures[0] if structures else None
    except Exception as exc:
        logger.debug("Failed to load %s: %s", cif_path, exc)
        return None


# ---------------------------------------------------------------------------
# Slab construction (conventional cell + ASE surface + supercell)
# ---------------------------------------------------------------------------

def _build_slab(
    bulk_structure,
    miller_index: Tuple[int, int, int],
    n_layers: int,
    supercell: Tuple[int, int],
):
    """Build a surface slab from *bulk_structure*.

    Steps:
    1. Convert to conventional standard cell via SpacegroupAnalyzer.
    2. Build slab with ASE's ``surface()`` (adds vacuum, orients surface ⊥ z).
    3. Enforce 3D PBC (required for pymatgen).
    4. Make a (*supercell[0]* × *supercell[1]* × 1) repeat.
    5. Convert to pymatgen Structure.

    Returns the pymatgen Structure or ``None`` on failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            analyzer = SpacegroupAnalyzer(bulk_structure, symprec=0.1)
            conventional = analyzer.get_conventional_standard_structure()

        atoms = AseAtomsAdaptor.get_atoms(conventional)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slab_atoms = ase_build_surface(atoms, miller_index, n_layers, vacuum=VACUUM)

        # Enforce 3D periodicity so pymatgen accepts it
        slab_atoms.pbc = [True, True, True]

        nx, ny = supercell
        if nx > 1 or ny > 1:
            slab_atoms = slab_atoms.repeat([nx, ny, 1])

        return AseAtomsAdaptor.get_structure(slab_atoms)

    except Exception as exc:
        logger.debug(
            "Slab build failed (miller=%s, layers=%d, supercell=%s): %s",
            miller_index, n_layers, supercell, exc,
        )
        return None


def _molecule_footprint_xy(mol: Molecule) -> Tuple[float, float]:
    """Return (width_x, width_y) of the molecule's bounding box in Å."""
    coords = np.array(mol.cart_coords)
    return (
        float(coords[:, 0].max() - coords[:, 0].min()),
        float(coords[:, 1].max() - coords[:, 1].min()),
    )


def _choose_supercell(
    slab,
    mol: Molecule,
    base_nx: int,
    base_ny: int,
    max_repeat: int = 4,
) -> Tuple[int, int]:
    """Return supercell repeat factors so the surface fits the molecule.

    Starts from (*base_nx*, *base_ny*) and bumps each dimension up if the
    molecule's footprint exceeds the corresponding cell vector length.
    Capped at *max_repeat* in each direction.
    """
    cell = slab.lattice.matrix  # 3×3
    a_len = float(np.linalg.norm(cell[0]))
    b_len = float(np.linalg.norm(cell[1]))

    mol_x, mol_y = _molecule_footprint_xy(mol)

    nx = max(base_nx, math.ceil((mol_x + 3.0) / a_len))
    ny = max(base_ny, math.ceil((mol_y + 3.0) / b_len))

    return min(nx, max_repeat), min(ny, max_repeat)


# ---------------------------------------------------------------------------
# Adsorbate placement
# ---------------------------------------------------------------------------

def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly distributed 3D rotation matrix via random quaternion."""
    # Shoemake (1992): uniform random quaternion from three uniform randoms
    u1, u2, u3 = rng.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1)     * np.sin(2 * np.pi * u3),
        np.sqrt(u1)     * np.cos(2 * np.pi * u3),
    ])  # [x, y, z, w]
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _rotate_molecule(mol: Molecule, R: np.ndarray) -> Molecule:
    """Return a new Molecule with Cartesian coords rotated by matrix *R*."""
    centroid = np.mean(mol.cart_coords, axis=0)
    new_coords = (mol.cart_coords - centroid) @ R.T + centroid
    return Molecule(mol.species, new_coords)


def _place_adsorbate(
    slab,
    molecule: Molecule,
    rng: np.random.Generator,
    height: float = ADSORPTION_HEIGHT,
    min_separation: float = MIN_SEPARATION,
    n_rotation_attempts: int = 6,
) -> Tuple[Optional[Any], Optional[List[int]]]:
    """Place *molecule* on *slab* with a random rotation and random XY position.

    Strategy
    --------
    For each candidate adsorption site (shuffled):
      - Apply a random 3D rotation to the molecule.
      - Shift the molecule centre to the site XY + a random fractional XY
        jitter within ±0.15 of the cell vectors, then check PBC-aware
        separations.

    Only placements that satisfy both slab–adsorbate (≥ ``min_separation``)
    and intra-adsorbate (≥ 0.5 Å) distance constraints are accepted.

    Returns ``(adsorbed_structure, molecule_indices)`` on success,
    ``(None, None)`` on failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            finder = AdsorbateSiteFinder(slab)
            ads_sites = finder.find_adsorption_sites(distance=height)

        all_sites = (
            ads_sites.get("ontop", [])
            + ads_sites.get("bridge", [])
            + ads_sites.get("hollow", [])
        )
        if not all_sites:
            return None, None

        site_arr = np.array(all_sites)
        n_slab = len(slab)
        lattice = slab.lattice

        for site_idx in rng.permutation(len(site_arr)):
            base_site = site_arr[site_idx]

            for _ in range(n_rotation_attempts):
                # --- random rotation ---
                R = _random_rotation_matrix(rng)
                rotated_mol = _rotate_molecule(molecule, R)

                # --- random XY jitter in fractional coords ---
                frac_jitter = rng.uniform(-0.15, 0.15, size=2)
                cart_jitter = (
                    frac_jitter[0] * lattice.matrix[0]
                    + frac_jitter[1] * lattice.matrix[1]
                )
                site = base_site + np.array([cart_jitter[0], cart_jitter[1], 0.0])

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        adsorbed = finder.add_adsorbate(rotated_mol, site)

                    n_ads = len(adsorbed) - n_slab
                    if n_ads <= 0:
                        continue

                    ads_indices = list(range(n_slab, len(adsorbed)))
                    dm = adsorbed.distance_matrix

                    # Slab–adsorbate separation (PBC already folded by pymatgen)
                    sub_mat = dm[np.ix_(ads_indices, list(range(n_slab)))]
                    if sub_mat.min() < min_separation:
                        continue

                    # Intra-adsorbate sanity (collapsed geometry check)
                    if len(ads_indices) > 1:
                        ads_sub = dm[np.ix_(ads_indices, ads_indices)]
                        np.fill_diagonal(ads_sub, np.inf)
                        if ads_sub.min() < 0.5:
                            continue

                    return adsorbed, ads_indices

                except Exception as exc:
                    logger.debug(
                        "Adsorbate placement error at site %d: %s", site_idx, exc
                    )
                    continue

    except Exception as exc:
        logger.debug("AdsorbateSiteFinder error: %s", exc)

    return None, None


# ---------------------------------------------------------------------------
# CIF serialisation
# ---------------------------------------------------------------------------

def _structure_to_cif(structure) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        writer = CifWriter(structure, symprec=None)
        return str(writer)


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------


class RemoveMoleculeDataGenerator(BaseDataGenerator):
    """Generate dataset items for the RemoveMoleculeAction active task.

    Parameters
    ----------
    cif_folder : str
        Path to a folder of bulk CIF files.
    zinc_path : str | None
        Path to the ZINC SMILES file (one SMILES per line).  Defaults to
        ``data/molecules/zinc.txt`` under the project root.
    miller_indices : list[tuple] | None
        Miller indices to try (in order) when cutting slabs.
    n_layers_range : tuple[int, int]
        (min, max) number of atomic layers in the slab (inclusive).
        A random value is drawn per sample.  Default: (2, 5).
    supercell_range : tuple[int, int]
        (min, max) repeat factor in each in-plane direction.  The actual
        supercell is additionally expanded so the molecule fits.
        Default: (1, 3).
    seed : int
        Random seed.
    max_attempts : int
        Retry budget per requested sample.
    allow_repeat_structures : bool
        Whether the same bulk CIF may be used for multiple samples.
    """

    def __init__(
        self,
        cif_folder: str,
        zinc_path: Optional[str] = None,
        miller_indices: Optional[List[Tuple[int, int, int]]] = None,
        n_layers_range: Tuple[int, int] = (1, 3),
        supercell_range: Tuple[int, int] = (1, 3),
        seed: Optional[int] = 75,
        max_attempts: int = 20,
        allow_repeat_structures: bool = True,
    ):
        super().__init__(seed=seed)
        self.cif_folder = cif_folder
        self.zinc_path = zinc_path or _DEFAULT_ZINC_PATH
        self.miller_indices = miller_indices or DEFAULT_MILLER_INDICES
        self.n_layers_range = n_layers_range
        self.supercell_range = supercell_range
        self.max_attempts = max_attempts
        self.allow_repeat_structures = allow_repeat_structures

        if not os.path.exists(cif_folder):
            raise ValueError(f"CIF folder not found: {cif_folder}")
        if not os.path.exists(self.zinc_path):
            raise FileNotFoundError(
                f"ZINC SMILES file not found: {self.zinc_path}\n"
                "Place zinc.txt at data/molecules/zinc.txt or pass zinc_path= explicitly."
            )

        self.cif_files = [
            os.path.join(cif_folder, f)
            for f in sorted(os.listdir(cif_folder))
            if f.endswith(".cif")
        ]
        if not self.cif_files:
            raise ValueError(f"No CIF files found in {cif_folder}")

        self._cif_indices = self._init_indices()
        self._zinc_smiles: Optional[List[str]] = None  # loaded lazily

    def _init_indices(self) -> List[int]:
        n = len(self.cif_files)
        indices = list(range(n))
        if not self.allow_repeat_structures:
            return self.rng.permutation(n).tolist()
        self.rng.shuffle(indices)
        return indices

    def _get_zinc_smiles(self) -> List[str]:
        if self._zinc_smiles is None:
            self._zinc_smiles = _load_zinc_smiles(self.zinc_path)
        return self._zinc_smiles

    def _try_generate_sample(self, cif_path: str) -> Optional[Dict[str, Any]]:
        mp_id = os.path.splitext(os.path.basename(cif_path))[0]

        bulk = _load_bulk_structure(cif_path)
        if bulk is None:
            return None

        # Random slab parameters
        n_layers = int(self.rng.integers(self.n_layers_range[0], self.n_layers_range[1] + 1))
        base_nx = int(self.rng.integers(self.supercell_range[0], self.supercell_range[1] + 1))
        base_ny = int(self.rng.integers(self.supercell_range[0], self.supercell_range[1] + 1))

        # Sample a ZINC molecule first (so we can size the supercell)
        zinc_smiles = self._get_zinc_smiles()
        smiles = zinc_smiles[int(self.rng.integers(len(zinc_smiles)))]
        pmg_mol = _build_molecule_from_smiles(smiles)
        if pmg_mol is None:
            return None

        # Build a base 1×1 slab to read off cell vector lengths for supercell sizing
        base_slab = None
        used_miller = None
        for mi in self.miller_indices:
            base_slab = _build_slab(bulk, mi, n_layers, supercell=(1, 1))
            if base_slab is not None:
                used_miller = list(mi)
                break
        if base_slab is None:
            return None

        nx, ny = _choose_supercell(base_slab, pmg_mol, base_nx, base_ny)

        # Rebuild with final supercell (skip rebuild if 1×1 already)
        if nx == 1 and ny == 1:
            slab = base_slab
        else:
            slab = _build_slab(bulk, tuple(used_miller), n_layers, supercell=(nx, ny))
            if slab is None:
                return None

        adsorbed, mol_indices = _place_adsorbate(slab, pmg_mol, self.rng)
        if adsorbed is None:
            return None

        slab_composition = {
            str(el): int(cnt)
            for el, cnt in slab.composition.as_dict().items()
        }

        try:
            input_cif = _structure_to_cif(adsorbed)
            output_cif = _structure_to_cif(slab)
        except Exception as exc:
            logger.debug("CIF serialisation failed: %s", exc)
            return None

        return {
            "action_type": "RemoveMoleculeAction",
            "task_category": "active",
            "problem_id": str(uuid.uuid4()),
            "mp_id": mp_id,
            "action_prompt": "Remove the molecule(s) inside the structure.",
            "input": input_cif,
            "output": output_cif,
            "verifiers": [
                "output_format",
                "cif_parsing",
                "atom_count",
                "structure_match",
            ],
            "metadata": {
                "molecule_smiles": smiles,
                "molecule_formula": pmg_mol.composition.formula.replace(" ", ""),
                "molecule_num_atoms": len(pmg_mol),
                "molecule_indices": mol_indices,
                "slab_composition": slab_composition,
                "miller_index": used_miller,
                "n_layers": n_layers,
                "supercell": [nx, ny],
            },
        }

    def generate(self, num_samples: int = -1, **kwargs) -> Iterator[Dict[str, Any]]:
        if num_samples < 0:
            num_samples = len(self._cif_indices)

        count = 0
        cif_cursor = 0

        while count < num_samples:
            generated = False

            for _ in range(self.max_attempts):
                if cif_cursor >= len(self._cif_indices):
                    if not self.allow_repeat_structures:
                        logger.info("No more unique structures available.")
                        return
                    self._cif_indices = self._init_indices()
                    cif_cursor = 0

                cif_path = self.cif_files[self._cif_indices[cif_cursor]]
                cif_cursor += 1

                sample = self._try_generate_sample(cif_path)
                if sample is not None:
                    yield sample
                    count += 1
                    generated = True
                    break

            if not generated:
                logger.warning(
                    "Could not generate sample %d after %d attempts.",
                    count + 1,
                    self.max_attempts,
                )
                break

    def generate_and_save(
        self,
        output_dir: str,
        num_samples: int,
        filename: str = "RemoveMoleculeAction.json",
    ) -> str:
        """Generate *num_samples* items and write them to *output_dir/filename*."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        samples: List[Dict[str, Any]] = []
        for item in self.generate(num_samples=num_samples):
            samples.append(item)
            if len(samples) % 10 == 0:
                logger.info("Generated %d / %d samples", len(samples), num_samples)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(samples, fh, indent=2, ensure_ascii=False)

        logger.info("Saved %d samples to %s", len(samples), output_path)
        return output_path

