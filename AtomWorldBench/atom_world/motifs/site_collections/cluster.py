"""Motif comprising multiple atoms."""
from typing import List, Optional

from ase import Atoms
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
) -> List[SiteMotif]:
    """Detect atoms in the given structure around fractional coordinates.

    Notice: all site motifs detected by this method will be set to the default name.
    Args:
        atoms(Atoms): The structure to analyze, represented as an ASE Atoms object.
        site_index (int): The index of the site around which to detect neighbors.
        cutoff (float): The radius within which to detect atoms. Default is 3.0.
        symbols (Optional[List[str]]): If provided, only atoms with these symbols will be detected.
            Default is None, which means no filtering.

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

    positions_valid = (
            atoms.get_positions(wrap=False)[indices_j_valid] +
            offsets_valid @ atoms.cell.complete()
    )
    symbols_valid = atoms.get_chemical_symbols()[indices_j_valid]
    charges_valid = atoms.get_initial_charges()[indices_j_valid]

    deduplicated_motifs = []
    for ii in range(len(symbols_valid)):
        motif = SiteMotif(
            symbols=[symbols_valid[ii]],
            positions=[positions_valid[ii]],
            cell=atoms.cell,
            pbc=atoms.pbc,
            charges=charges_valid[ii],
            name=None,  # Default name will be generated in the SiteMotif class.
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
            in_atoms: Atoms,
            indices: List[int],
            offsets: Optional[ArrayLike] = None,
            name: Optional[str] = None,
            allow_translation_equivalence: Optional[bool] = None,
    ):
        """ClusterMotif constructor.

        Args:
            in_atoms (Atoms): The ASE Atoms object to create the motif from.
            indices (list of int): Original indices from structure.
                Indices should always be provided, as the motif belongs to a specific structure.
            offsets (ArrayLike, optional): The cell offsets for each atom in the motif.
                Cell offsets are the integer part of the fractional coordinates in the form of
                triplets (i, j, k). If None, will assume all zeros.
            name (str, optional): Human-readable motif name. Optional.
             If None, will generate a default name.
            allow_translation_equivalence (bool):
                If True, the motif can be considered equivalent to another motif
                if they are related by an integer translation.
                Default is not given, then will use the global setting ALLOW_TRANSLATION_EQUIVALENCE.
        """
        super().__init__(
            in_atoms,
            indices,
            offsets,
            name,
            allow_translation_equivalence,
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
    def detect_random_one(
            cls,
            atoms: Atoms,
            cluster_size: int = 3,
            max_cluster_radius: float = 3.0,
            n_attempts: int = 10,
            randomize_symbols: bool = False,
            seed: Optional[int] = None,
            allow_translation_equivalence: Optional[bool] = None,
    ) -> 'ClusterMotif':
        """Detect a random cluster motif from the given Atoms object.

        Args:
            atoms (Atoms): The Atoms object from which to detect the cluster.
            cluster_size (int): The desired size of the cluster. Defaults to 3.
            max_cluster_radius (float): Maximum allowable radius of the cluster.
                Defaults to 3.0.
            n_attempts (int): Number of attempts to find a valid cluster. Defaults to 10.
            randomize_symbols (bool): If True, the symbols of the atoms in the motif will be
                randomly chosen from the symbols of the atoms in the provided Atoms object.
                If False, the symbols will be set to None, meaning all atoms in the region
                will be included regardless of their symbols.
                Defaults to False.
            seed (Optional[int]): Random seed for reproducibility. Defaults to None.
            allow_translation_equivalence (Optional[bool]): If True, the motif can be considered
                equivalent to another motif if they are related by an integer translation.
                Default is not given, then will use the global setting ALLOW_TRANSLATION_EQUIVALENCE.

        Returns:
            ClusterMotif: A ClusterMotif instance representing the detected cluster.
        """
        rng = np.random.default_rng(seed)
        if randomize_symbols:
            all_symbols = list(set(atoms.get_chemical_symbols()))
            num_symbols = int(rng.integers(1, len(all_symbols) + 1))
            symbols = rng.choice(
                all_symbols, size=num_symbols, replace=False
            ).tolist()
        else:
            symbols = None


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
                        (existing_cluster + site).radius <= max_cluster_radius
                )
            ]

        def _detect_attempt(a) -> ClusterMotif | None:
            # Perform a single detection attempt.
            rand_idx = int(rng.integers(0, len(a)))
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
