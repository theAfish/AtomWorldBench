"""Motif comprising multiple atoms."""
from typing import List, Optional
from copy import deepcopy

from ase import Atoms
from ase.data import chemical_symbols
import numpy as np
from numpy.typing import ArrayLike

from .base import BaseSiteCollectionMotif
from ..base import BaseMotif
from .site import SiteMotif

from ....common.registry import register
from ....utils.neighbor_utils import detect_indices_offsets_around_frac_coords


def detect_neighbor_sites_around_site_index(
        atoms: Atoms,
        site_index: int,
        cutoff: float = 3.0,
        symbols: Optional[List[str]] = None,
        allow_translation_equivalence: Optional[bool] = None,
) -> List[SiteMotif]:
    """Detect atoms in the given structure around fractional coordinates.

    Notice: all site motifs detected by this method will be set to the default name.
    Args:
        atoms(Atoms): The structure to analyze, represented as an ASE Atoms object.
        site_index (int): The index of the site around which to detect neighbors.
        cutoff (float): The radius within which to detect atoms. Default is 3.0.
        symbols (Optional[List[str]]): If provided, only atoms with these symbols will be detected.
            Default is None, which means no filtering.
        allow_translation_equivalence (bool): If True, the detected motifs can be considered
            equivalent to another motif if they are related by an integer translation.
            Default is set in the global setting ALLOW_TRANSLATION_EQUIVALENCE.

    Returns:
        List of detected site motifs within the specified radius.
    """
    if "X" in atoms.get_chemical_symbols():
        raise ValueError("The structure already contains a dummy atom with symbol 'X'. "
                         "Please use a different symbol for the dummy atom.")
    frac_coords = atoms.get_scaled_positions(wrap=False)[site_index]

    indices_j_valid, offsets_valid = detect_indices_offsets_around_frac_coords(
        atoms, frac_coords, cutoff, symbols=symbols
    )
    # Exclude the provided site itself from the results.
    indices_j_valid = np.array(indices_j_valid)
    offsets_valid = offsets_valid[indices_j_valid != site_index]
    indices_j_valid = indices_j_valid[indices_j_valid != site_index]

    deduplicated_motifs = []
    for ii in range(len(indices_j_valid)):
        motif = SiteMotif(
            in_atoms=atoms,
            offsets=[offsets_valid[ii]],
            name=None,  # Use default name for detected sites.
            allow_translation_equivalence=allow_translation_equivalence,  # Use global setting.
            indices=[int(indices_j_valid[ii])]  # Ensure indices are integers.,
        )
        deduplicated_motifs.append(motif)
    return deduplicated_motifs


@register(BaseMotif, aliases=["cluster", "atom-cluster"])
@register(BaseSiteCollectionMotif, aliases=["cluster", "atom-cluster"])
class ClusterMotif(BaseSiteCollectionMotif):
    """Motif representing a cluster of atoms.

    This motif is defined by a list of species and their fractional coordinates.
    It can be used to represent clusters of atoms in a structure.
    """
    def __init__(
            self,
            in_atoms: Optional[Atoms] = None,
            indices: Optional[List[int]] = None,
            offsets: Optional[ArrayLike] = None,
            atoms: Optional[Atoms] = None,
            name: Optional[str] = None,
            allow_translation_equivalence: Optional[bool] = None,
    ):
        """ClusterMotif constructor.

        Args:
            in_atoms (Atoms, optional): The ASE Atoms object to create the motif from.
                Notice: this object will always be wrapped at init if not already!
                All cell offsets will be computed relative to the wrapped positions.
            indices (list of int, optional): Original indices from structure.
                Indices should always be provided, as the motif belongs to a specific structure.
            offsets (ArrayLike, optional): The cell offsets for each atom in the motif.
                Cell offsets are the integer part of the fractional coordinates in the form of
                triplets (i, j, k) representing their unwrapped location in periodic images.
                If None, will assume all zeros.
            atoms (Atoms, optional): An ASE Atoms object representing the motif.
                When none of in_atoms, indices, offsets are provided, and atoms is provided,
                will create a motif directly from atoms. In this case, the motif can only be
                added in the AddMotifAction.
            name (str, optional): Human-readable motif name. Optional.
             If None, will generate a default name.
            allow_translation_equivalence (bool):
                If True, the motif can be considered equivalent to another motif
                if they are related by an integer translation.
                Default is not given, then will use the global setting ALLOW_TRANSLATION_EQUIVALENCE.
        """
        BaseSiteCollectionMotif.__init__(
            self,
            in_atoms=in_atoms,
            indices=indices,
            offsets=offsets,
            atoms=atoms,
            name=name,
            allow_translation_equivalence=allow_translation_equivalence,
        )

    def __post_init__(self):
        # Widen restriction to allow single point cluster, such that cluster detector can grow from it.
        if len(self) < 1:
            raise ValueError("ClusterMotif must contain at least one atom.")

    def _get_default_name(self) -> str:
        """Generate a default name for the cluster motif based on species and coordinates."""
        if len(self) == 1:
            prefix = "point"
        elif len(self) == 2:
            prefix = "pair"
        elif len(self) == 3:
            prefix = "triplet"
        elif len(self) == 4:
            prefix = "quadruplet"
        elif len(self) == 5:
            prefix = "quintuplet"
        elif len(self) == 6:
            prefix = "sextuplet"
        else:
            prefix = f"{len(self)}-sites cluster"

        return f"a {prefix} of atoms/species {', '.join(map(str, self.species_strings))}"

    @property
    def site_motifs(self) -> List[SiteMotif]:
        """Return a list of SiteMotif objects representing the sites in the cluster."""
        return [
            SiteMotif(
                self.in_atoms,
                indices=[self.indices[i]],
                offsets=[self.cell_offsets[i]],
                allow_translation_equivalence=self.allow_translation_equivalence
            )  # Use default name for sites.
            for i in range(len(self))
        ]


    @classmethod
    def _detect_one_non_additive(
            cls,
            atoms: Atoms,
            cluster_size: int = 3,
            max_cluster_radius: float = 3.0,
            n_attempts: int = 50,
            randomize_symbols: bool = False,
            seed: Optional[int] = None,
            allow_translation_equivalence: Optional[bool] = None,
            excluded_site_indices: Optional[List[int]] = None,
    ) -> 'ClusterMotif':
        """Detect a random cluster motif from the given Atoms object.

        Used when not in additive mode.
        """
        atoms.wrap()
        rng = np.random.default_rng(seed)
        if randomize_symbols:
            all_symbols = list(set(atoms.get_chemical_symbols()))
            num_symbols = int(rng.integers(1, len(all_symbols) + 1))
            symbols = rng.choice(
                all_symbols, size=num_symbols, replace=False
            ).tolist()
        else:
            symbols = None

        excluded_site_indices = excluded_site_indices or []

        def _filter_neighbors(
                existing_cluster,
                neighbor_site_motifs,
        ):
            # Filter out neighbors that are already in the existing cluster, or
            # exceeds the maximum cluster radius when added to the existing cluster.
            return [
                site for site in neighbor_site_motifs
                if (
                        site not in existing_cluster.site_motifs and
                        (existing_cluster + site).radius <= max_cluster_radius and
                        site.indices[0] not in excluded_site_indices  # Exclude specified indices.
                )
            ]

        def _detect_attempt(a) -> ClusterMotif | None:
            # Perform a single detection attempt.
            allowed_indices = (
                list(set(range(len(a))) - set(excluded_site_indices))
            ) if excluded_site_indices else list(range(len(a)))
            rand_idx = int(rng.choice(allowed_indices))
            rand_indices = [rand_idx]
            c = ClusterMotif(
                a, indices=[rand_idx], allow_translation_equivalence=allow_translation_equivalence
            )
            for _ in range(cluster_size - 1):
                # Randomly select a site to grow the cluster around.
                neighbor_site_motifs = detect_neighbor_sites_around_site_index(
                    a,
                    rand_indices[-1],
                    cutoff=max_cluster_radius,
                    symbols=symbols
                )
                deduplicated_site_motifs = _filter_neighbors(c, neighbor_site_motifs)
                if len(deduplicated_site_motifs) == 0:
                    return None  # Failed to grow the cluster.
                deduplicated_site_indices = [
                    int(site.indices[0]) for site in deduplicated_site_motifs
                ]
                rand_nn_idx = int(rng.integers(0, len(deduplicated_site_indices)))
                c += deduplicated_site_motifs[rand_nn_idx]
                rand_indices.append(deduplicated_site_indices[rand_nn_idx])
            return c

        for _ in range(n_attempts):
            cluster = _detect_attempt(atoms)
            if cluster is not None:
                return cluster

        raise RuntimeError(
            f"Failed to detect a valid {cls.__name__} of size {cluster_size}"
            f" within radius {max_cluster_radius} in {n_attempts} attempts."
            f" Please check the structure and cutoff parameters."
        )

    @classmethod
    def _detect_one_additive(
            cls,
            atoms: Atoms,
            allowed_symbols: List[str],
            cluster_size: int = 3,
            max_cluster_radius: float = 3.0,
            seed: Optional[int] = None,
    ) -> 'ClusterMotif':
        """Build a random additive cluster motif.

        Used when in additive mode.
        """
        rng = np.random.default_rng(seed)
        selected_symbols = rng.choice(
            allowed_symbols, size=cluster_size, replace=True
        ).tolist()
        # Generate random positions within a sphere of radius max_cluster_radius.
        positions = [np.array([0.0, 0.0, 0.0])]
        while len(positions) < cluster_size:
            # Draw a random point uniformly within a sphere.
            r = rng.uniform(0, max_cluster_radius)
            theta = rng.uniform(0, np.pi)
            phi = rng.uniform(0, 2 * np.pi)
            dx = np.array(
                [r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)]
            )
            x = positions[-1] + dx
            # Check if the new position is within the max_cluster_radius from all previous points.
            if all(np.linalg.norm(x - pos) <= max_cluster_radius for pos in positions):
                positions.append(x)
        positions = np.array(positions)

        # Create a dummy Atoms object to hold the cluster motif.
        # The cell copies from atoms. Typically,
        # cell does not matter, only centroid matters.
        cluster_atoms = Atoms(
            symbols=selected_symbols,
            positions=positions.tolist(),
            cell=atoms.cell,
            pbc=atoms.pbc,
        )

        # Initialize as additive.
        return cls(
            in_atoms=None,
            indices=None,
            offsets=None,
            atoms=cluster_atoms,
            name=None,
            allow_translation_equivalence=None,
        )

    @classmethod
    def detect_random_one(
            cls,
            atoms: Atoms,
            additive_mode: bool = False,
            additive_mode_allowed_symbols: Optional[List[str]] = None,
            cluster_size: int = 3,
            max_cluster_radius: float = 3.0,
            n_attempts: int = 50,
            randomize_symbols: bool = False,
            seed: Optional[int] = None,
            allow_translation_equivalence: Optional[bool] = None,
            excluded_site_indices: Optional[List[int]] = None,
    ) -> 'ClusterMotif':
        """Detect a random cluster motif from the given Atoms object.

        Args:
            atoms (Atoms): The Atoms object from which to detect the cluster.
                Notice: this object will always be wrapped at init if not already!
                All cell offsets will be computed relative to the wrapped positions.
            additive_mode (bool): If True, the detected motif will be in additive mode,
                meaning it is not tied to specific atoms in the structure.
                Instead, it represents a generic cluster that can be placed anywhere.
                If False, the motif will be tied to specific atoms in the provided Atoms object.
                Defaults to False.
            additive_mode_allowed_symbols (Optional[List[str]]): When in additive mode,
                the list of allowed chemical symbols for the atoms in the cluster motif.
                If None, all chemical symbols from the periodic table will be considered,
                or choose random element symbol if randomize_symbols is True.
                This parameter is ignored when not in additive mode.
            cluster_size (int): The desired size of the cluster. Defaults to 3.
            max_cluster_radius (float): Maximum allowable radius of the cluster.
                Defaults to 3.0.
            n_attempts (int): Number of attempts to find a valid cluster. Defaults to 50.
            randomize_symbols (bool): If True, the symbols of the atoms in the motif will be
                randomly chosen from the symbols of the atoms in the provided Atoms object.
                If False, the symbols will be set to None, meaning all atoms in the region
                will be included regardless of their symbols.
                Defaults to False.
            seed (Optional[int]): Random seed for reproducibility. Defaults to None.
            allow_translation_equivalence (Optional[bool]): If True, the motif can be considered
                equivalent to another motif if they are related by an integer translation.
                Default is not given, then will use the global setting ALLOW_TRANSLATION_EQUIVALENCE.
            excluded_site_indices (Optional[List[int]]): List of site indices to exclude from selection.
                Used to prevent overlap, if already generated motifs, should not be selected again.
                Default is None.

        Returns:
            ClusterMotif: A ClusterMotif instance representing the detected cluster.
        """
        if additive_mode:
            if additive_mode_allowed_symbols is None:
                rng = np.random.default_rng(seed)
                all_symbols = deepcopy(chemical_symbols[1:])  # Exclude the first entry which is ''
                if randomize_symbols:
                    num_symbols = int(rng.integers(1, len(all_symbols) + 1))
                    allowed_symbols = rng.choice(
                        all_symbols, size=num_symbols, replace=False
                    ).tolist()
                else:
                    allowed_symbols = all_symbols
            else:
                allowed_symbols = additive_mode_allowed_symbols
            return cls._detect_one_additive(
                atoms,
                allowed_symbols=allowed_symbols,
                cluster_size=cluster_size,
                max_cluster_radius=max_cluster_radius,
                seed=seed,
            )
        else:
            return cls._detect_one_non_additive(
                atoms,
                cluster_size=cluster_size,
                max_cluster_radius=max_cluster_radius,
                n_attempts=n_attempts,
                randomize_symbols=randomize_symbols,
                seed=seed,
                allow_translation_equivalence=allow_translation_equivalence,
                excluded_site_indices=excluded_site_indices,
            )
