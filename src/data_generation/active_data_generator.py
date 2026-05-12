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
#   src/data_generation/ → molecules/zinc.txt
_DEFAULT_ZINC_PATH: str = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "molecules", "zinc.txt")
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

# Bulk-insertion variant constants
BULK_MIN_SEPARATION: float = 1.8      # Å — bulk atoms within this of any mol atom are removed;
                                      #     also the minimum distance required to remaining atoms
BULK_MAX_REMOVED_FRACTION: float = 0.25  # cap on fraction of bulk atoms that may be removed
BULK_CLEARANCE: float = 2.0           # Å — extra clearance when sizing the supercell

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
# Bulk-insertion helpers (new variant)
# ---------------------------------------------------------------------------

def _molecule_max_extent(mol: Molecule) -> float:
    """Return the maximum distance from the molecular centroid to any atom (Å)."""
    coords = np.array(mol.cart_coords)
    centroid = coords.mean(axis=0)
    return float(np.max(np.linalg.norm(coords - centroid, axis=1)))


def _build_bulk_supercell(bulk_structure, repeat: int):
    """Build an isotropic (repeat × repeat × repeat) supercell of *bulk_structure*.

    The conventional standard cell is used as the starting point.
    Returns the pymatgen Structure or ``None`` on failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            analyzer = SpacegroupAnalyzer(bulk_structure, symprec=0.1)
            conventional = analyzer.get_conventional_standard_structure()
        supercell = conventional.copy()
        if repeat > 1:
            supercell.make_supercell([repeat, repeat, repeat])
        return supercell
    except Exception as exc:
        logger.debug("Bulk supercell build failed (repeat=%d): %s", repeat, exc)
        return None


def _choose_bulk_repeat(bulk_structure, mol: Molecule, base_repeat: int, max_repeat: int = 3) -> int:
    """Return the isotropic repeat factor so the supercell comfortably contains *mol*.

    Each lattice vector must be at least ``2 * (mol_extent + BULK_CLEARANCE)`` Å.
    """
    extent = _molecule_max_extent(mol)
    min_cell_len = 2.0 * (extent + BULK_CLEARANCE)
    try:
        cell = bulk_structure.lattice.matrix
        min_axis = min(float(np.linalg.norm(cell[i])) for i in range(3))
    except Exception:
        return base_repeat
    if min_axis < 1e-6:
        return base_repeat
    needed = math.ceil(min_cell_len / max(min_axis, 1e-6))
    return min(max(base_repeat, needed), max_repeat)


def _insert_molecule_in_bulk(
    bulk_supercell,
    molecule: Molecule,
    rng: np.random.Generator,
    min_separation: float = BULK_MIN_SEPARATION,
    max_removed_fraction: float = BULK_MAX_REMOVED_FRACTION,
    n_attempts: int = 40,
) -> Tuple[Optional[Any], Optional[Any], Optional[List[int]], Optional[Dict[str, int]], Optional[int]]:
    """Insert *molecule* into *bulk_supercell*, carving out overlapping bulk atoms.

    For each attempt a fresh random rotation **and** random position inside the
    supercell are drawn.  Bulk atoms within *overlap_threshold* of any molecule
    atom (under PBC minimum-image) are removed to make room.

    Returns
    -------
    (input_structure, output_structure, mol_indices, bulk_composition, n_removed)
        ``input_structure``  — bulk (cavities carved) + molecule
        ``output_structure`` — bulk (cavities carved), no molecule
        ``mol_indices``      — 0-based indices of molecule atoms in *input_structure*
        ``bulk_composition`` — elemental composition of *output_structure*
        ``n_removed``        — number of bulk atoms deleted
        All ``None`` on failure.
    """
    lattice = bulk_supercell.lattice
    n_bulk = len(bulk_supercell)
    max_removed = max(1, int(n_bulk * max_removed_fraction))

    # Pre-compute bulk fractional coords once
    bulk_frac_coords = bulk_supercell.frac_coords  # (n_bulk, 3)

    for _ in range(n_attempts):
        # Random rotation for the molecule
        R = _random_rotation_matrix(rng)
        rotated_mol = _rotate_molecule(molecule, R)

        # Centre the molecule at the origin, then shift to a random cell position.
        # Avoid fractional positions too close to cell boundaries (0.1–0.9) so
        # the molecule stays well inside and doesn't straddle a PBC face.
        mol_cart_centered = rotated_mol.cart_coords - rotated_mol.cart_coords.mean(axis=0)
        frac_center = rng.uniform(0.15, 0.85, size=3)
        cart_center = lattice.get_cartesian_coords(frac_center)
        mol_cart_coords = mol_cart_centered + cart_center  # (n_mol, 3)

        try:
            mol_frac_coords = lattice.get_fractional_coords(mol_cart_coords)  # (n_mol, 3)

            # Vectorised PBC distances: (n_mol, n_bulk)
            diff = bulk_frac_coords[np.newaxis, :, :] - mol_frac_coords[:, np.newaxis, :]
            diff -= np.round(diff)                        # minimum image
            cart_diff = diff @ lattice.matrix             # (n_mol, n_bulk, 3)
            all_dists = np.linalg.norm(cart_diff, axis=-1)  # (n_mol, n_bulk)

            # Bulk atoms to remove: within min_separation of ANY mol atom
            close_bulk_mask = (all_dists < min_separation).any(axis=0)  # (n_bulk,)
            close_bulk_indices = set(np.where(close_bulk_mask)[0].tolist())

            # Reject if too many bulk atoms would be carved out
            if len(close_bulk_indices) > max_removed:
                continue

            # Minimum distance from any mol atom to any REMAINING bulk atom
            remaining_dists = all_dists[:, ~close_bulk_mask]  # (n_mol, n_remaining)
            if remaining_dists.size > 0 and remaining_dists.min() < min_separation:
                continue

            # Intra-molecule geometry sanity (detect collapsed/overlapping atoms)
            if len(mol_cart_coords) > 1:
                diff_mat = mol_cart_coords[:, np.newaxis, :] - mol_cart_coords[np.newaxis, :, :]
                intra_dists = np.linalg.norm(diff_mat, axis=-1)
                np.fill_diagonal(intra_dists, np.inf)
                if intra_dists.min() < 0.5:
                    continue

            # Build output structure (bulk with close atoms removed)
            output_structure = bulk_supercell.copy()
            for idx in sorted(close_bulk_indices, reverse=True):
                output_structure.remove_sites([idx])

            # Build input structure (output + molecule appended)
            input_structure = output_structure.copy()
            mol_start_idx = len(input_structure)
            for spec, cart_coord in zip(rotated_mol.species, mol_cart_coords):
                input_structure.append(
                    spec, cart_coord, coords_are_cartesian=True, validate_proximity=False
                )
            mol_indices = list(range(mol_start_idx, len(input_structure)))

            bulk_composition = {
                str(el): int(cnt)
                for el, cnt in output_structure.composition.as_dict().items()
            }

            return input_structure, output_structure, mol_indices, bulk_composition, len(close_bulk_indices)

        except Exception as exc:
            logger.debug("Bulk insertion attempt failed: %s", exc)
            continue

    return None, None, None, None, None


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
        ``src/data_generation/molecules/zinc.txt`` (bundled with the package).
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
                "Place zinc.txt at src/data_generation/molecules/zinc.txt or pass zinc_path= explicitly."
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

    def _try_generate_sample_surface(self, cif_path: str) -> Optional[Dict[str, Any]]:
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
                "task_variant": "surface",
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

    def _try_generate_sample_bulk(self, cif_path: str) -> Optional[Dict[str, Any]]:
        """Bulk-interstitial variant: insert a molecule inside a 3-D periodic supercell.

        Pipeline
        --------
        1. Load bulk CIF → conventional standard cell.
        2. Build an isotropic supercell sized to fit the molecule.
        3. Randomly rotate and position the molecule inside the supercell.
        4. Remove bulk atoms that overlap with the molecule (PBC-aware).
        5. Input  = cavitated bulk + molecule.
           Output = cavitated bulk (molecule absent).
        """
        mp_id = os.path.splitext(os.path.basename(cif_path))[0]

        bulk = _load_bulk_structure(cif_path)
        if bulk is None:
            return None

        # Sample a ZINC molecule
        zinc_smiles = self._get_zinc_smiles()
        smiles = zinc_smiles[int(self.rng.integers(len(zinc_smiles)))]
        pmg_mol = _build_molecule_from_smiles(smiles)
        if pmg_mol is None:
            return None

        # Choose supercell repeat so the molecule fits comfortably
        base_repeat = int(self.rng.integers(self.supercell_range[0], self.supercell_range[1] + 1))
        repeat = _choose_bulk_repeat(bulk, pmg_mol, base_repeat)

        supercell = _build_bulk_supercell(bulk, repeat)
        if supercell is None:
            return None

        (
            input_struct,
            output_struct,
            mol_indices,
            bulk_composition,
            n_removed,
        ) = _insert_molecule_in_bulk(supercell, pmg_mol, self.rng)
        if input_struct is None:
            return None

        try:
            input_cif = _structure_to_cif(input_struct)
            output_cif = _structure_to_cif(output_struct)
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
                "task_variant": "bulk",
                "molecule_smiles": smiles,
                "molecule_formula": pmg_mol.composition.formula.replace(" ", ""),
                "molecule_num_atoms": len(pmg_mol),
                "molecule_indices": mol_indices,
                "bulk_composition": bulk_composition,
                "n_removed_atoms": n_removed,
                "supercell_repeat": repeat,
            },
        }

    def _try_generate_sample(self, cif_path: str) -> Optional[Dict[str, Any]]:
        """Randomly dispatch between the surface-adsorption and bulk-interstitial variants."""
        if self.rng.integers(2) == 0:
            return self._try_generate_sample_surface(cif_path)
        return self._try_generate_sample_bulk(cif_path)

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

