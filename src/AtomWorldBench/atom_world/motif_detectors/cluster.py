from typing import List, Optional, Dict, Tuple
from copy import deepcopy

from ase import Atoms
from numpy.typing import ArrayLike
import numpy as np

from .base import BaseDetector
from .site import SiteDetector
from ..motifs import BaseMotif
from ..motifs.site import SiteMotif
from ..motifs.cluster import ClusterMotif


def grow_cluster(
        cluster: ClusterMotif,
        available_neighbors: List[SiteMotif],
        max_cluster_radius: float,
) -> Tuple[List[ClusterMotif], List[List[SiteMotif]]]:
    """Grow a cluster by adding available sites to the root cluster.

    Args:
        cluster (ClusterMotif): The current cluster to grow.
        available_neighbors (List[SiteMotif]):
         List of available neighboring sites to consider for growth.
        max_cluster_radius (float): Maximum allowed radius for the cluster.
    Returns:
        List[ClusterMotif]: A list of clusters that have been grown from the root cluster.
    """
    clusters = []
    remaining_neighbors = []
    for nid, site in enumerate(available_neighbors):
        new_cluster = cluster.copy()
        new_cluster += site
        if new_cluster.radius <= max_cluster_radius:
            # Name reset to default by ClusterMotif.__iadd__. Need to reassign if needed.
            clusters.append(new_cluster)
            remaining_neighbors.append(available_neighbors[: nid] + available_neighbors[nid + 1:])
    return clusters, remaining_neighbors


def deduplicate_clusters(
        clusters: List[ClusterMotif],
        remaining_neighbors: List[List[SiteMotif]],
        existing_clusters: List[ClusterMotif],
) -> Tuple[List[ClusterMotif], List[List[SiteMotif]]]:
    """Remove duplicate clusters from the list of clusters.

    Args:
        clusters (List[ClusterMotif]): List of clusters to check for duplicates.
        remaining_neighbors (List[List[SiteMotif]]): List of remaining neighbors corresponding to each cluster.
        existing_clusters (List[ClusterMotif]): List of already existing clusters to compare against.
    Returns:
        Tuple[List[ClusterMotif], List[List[SiteMotif]]]:
         A tuple containing the filtered list of clusters and their corresponding remaining neighbors.
    """
    unique_clusters = []
    unique_remaining_neighbors = []
    for cluster, neighbors in zip(clusters, remaining_neighbors):
        is_unique = True
        for existing_cluster in existing_clusters:
            if cluster == existing_cluster:
                is_unique = False
                break
        if is_unique:
            unique_clusters.append(cluster)
            unique_remaining_neighbors.append(neighbors)
    return unique_clusters, unique_remaining_neighbors


def deduplicate_same_list_clusters(
        clusters: List[ClusterMotif],
) -> List[ClusterMotif]:
    """Remove duplicate clusters from the list of clusters.

    Args:
        clusters (List[ClusterMotif]): List of clusters to check for duplicates.
    Returns:
        List[ClusterMotif]: A list of unique clusters.
    """
    unique_clusters = []
    for cluster in clusters:
        if cluster not in unique_clusters:
            unique_clusters.append(cluster)
    return unique_clusters


class ClusterDetector(BaseDetector):
    """Detects clusters of atoms in a structure.

    This detector identifies clusters based on the proximity of atoms and their species.
    It can be used to find groups of atoms that are close together and share similar properties.
    """
    def __init__(
            self,
            cutoff: float | Dict = 3.0,
            max_cluster_size: int = 2,
            max_cluster_radius: Optional[float] = None,
            must_include_center: bool = True,
            symbols: Optional[List[str]] = None,
            seed: Optional[int] = None,
    ):
        """Initialize the ClusterDetector with a default radius.

        Clusters will only be returned if they meet the specified criteria:
         1, Fewer sites than the specified `cluster_size`
         2, Radius less than or equal to `max_cluster_radius`.
         3, If `must_include_center` is True, the center atom must be part of the cluster.
         4, If `species_to_include` is specified, the cluster must contain only the specified species.

        Args:
            cutoff (float | Dict):
             The default radius for detecting clusters around fractional coordinates.
             If a float, it is used as the cutoff distance for all atoms.
             If a dict, it should map species to their respective cutoff distances.
              (same as in ASE's neighbor_list).
             Default is 3.0.
            max_cluster_size (int):
             Maximum number of atoms in each detected cluster. Default is 2 (doublet).
            max_cluster_radius (Optional[float]):
             Maximum allowed cluster radius for detection.
             Clusters larger than this radius will not be detected.
             Default is None, will be set to the max value in `cutoff` if not provided.
            must_include_center (bool):
             If True, the center atom must be included in the detected cluster. Default is True.
            symbols (Optional[List[str]]): List of species to include in the detected clusters.
             If None, all species are included. Default is None.
            seed (Optional[int]):
             Random seed for reproducibility of the `detect_one` method.
             Default is None, will generate with a random seed.
        """
        super().__init__(seed=seed)
        self.cutoff = cutoff
        self.max_cluster_size = max_cluster_size
        max_cutoff = cutoff if isinstance(cutoff, (float, int)) else max(cutoff.values())
        self.max_cluster_radius = max_cluster_radius if max_cluster_radius is not None else max_cutoff
        self.must_include_center = must_include_center
        self.symbols = symbols

    def detect_around_frac_coords(
            self,
            atoms: Atoms,
            frac_coords: ArrayLike,
    ) -> List[ClusterMotif]:
        """Detect clusters in the given structure around fractional coordinates.

        First detects all atoms within the specified radius from the given fractional coordinates,
        then groups them into clusters based on their proximity.

        Args:
            atoms (Atoms): The structure to analyze, represented as an ASE Atoms object.
                The structure should contain the atoms to be analyzed.
            frac_coords (ArrayLike): Fractional coordinates for the cluster detection center.
                Must be a one-dimensional array of shape (3,).

        Returns:
            List of all detected clusters with in the specified radius and size at least 2, less than
             or equal to `max_cluster_size`.
        """
        site_motifs = SiteDetector(
            cutoff=self.cutoff,
            symbols=self.symbols
        ).detect_around_frac_coords(
            atoms,
            frac_coords
        )

        if not self.must_include_center:
            empty_cluster = ClusterMotif(
                symbols=[],
                positions=[],
                cell=atoms.cell,
            )
            current_clusters = [empty_cluster]
            current_available_neighbors = [deepcopy(site_motifs)]
            saved_clusters = {0: [empty_cluster]}
        else:
            dists = [np.linalg.norm(site.cart_coords - frac_coords @ atoms.cell) for site in site_motifs]
            imin = np.argmin(dists)
            center_site = site_motifs[imin]
            init_cluster = ClusterMotif(
                symbols=center_site.get_chemical_symbols(),
                positions=center_site.get_positions(wrap=False),
                cell=center_site.get_cell(complete=True),
                pbc=center_site.get_cell(complete=True),
                charges=center_site.get_initial_charges(),
                name=center_site.name,
                indices=center_site.indices
            )
            current_clusters = [init_cluster]
            current_available_neighbors = [site_motifs[:imin] + site_motifs[imin + 1:]]
            saved_clusters = {len(init_cluster): [init_cluster]}

        while len(current_clusters[0]) < self.max_cluster_size and len(current_available_neighbors[0]) > 0:
            new_clusters = []
            new_available_neighbors = []
            for cluster, neighbors in zip(current_clusters, current_available_neighbors):
                clusters, remaining_neighbors = grow_cluster(
                    cluster,
                    neighbors,
                    self.max_cluster_radius
                )
                clusters, remaining_neighbors = deduplicate_clusters(
                    clusters,
                    remaining_neighbors,
                    new_clusters,
                )
                new_clusters.extend(clusters)
                new_available_neighbors.extend(remaining_neighbors)
            saved_clusters[len(new_clusters[0])] = new_clusters
            current_clusters = new_clusters
            current_available_neighbors = new_available_neighbors

        # Drop the empty cluster and point clusters.
        if 0 in saved_clusters:
            _ = saved_clusters.pop(0, None)
        if 1 in saved_clusters:
            _ = saved_clusters.pop(1, None)
        return [cluster for clusters in saved_clusters.values() for cluster in clusters]

    def detect_all(
            self,
            atoms: Atoms,
    ) -> List[ClusterMotif]:
        """Detect all clusters in the given structure.

        This method is used to detect all clusters in the structure.
        It uses the `detect_around_frac_coords` method to find clusters around each atom.
        Then deduplicates the clusters to ensure unique motifs.

        Args:
            atoms (Atoms): The structure to analyze, represented as an ASE Atoms object.

        Returns:
            List of all detected cluster motifs in the structure.
        """
        return deduplicate_same_list_clusters(
            super().detect_all(atoms)
        )

    def _get_symbol_valid_indices(self, atoms: Atoms) -> List[int]:
        """Get indices of atoms that match the specified symbols."""
        if self.symbols is None:
            return list(range(len(atoms)))
        else:
            return [
                i for i, symbol in enumerate(atoms.get_chemical_symbols())
                if symbol in self.symbols
            ]

    def detect_one(
            self,
            atoms: Atoms,
            size: Optional[int] = None,
            n_attempts: Optional[int] = 10,
    ) -> BaseMotif | None:
        """Detect a single cluster in the given structure.

        This method is used to detect a random single cluster in the structure.
        It uses the `detect_around_frac_coords` method to find clusters around the center atom.

        Args:
            atoms (Atoms): The structure to analyze, represented as an ASE Atoms object.
            size (Optional[int]): The size of the cluster to detect.
            If None, choose a size between 2 and `max_cluster_size`. Default is None.
            n_attempts (Optional[int]): The number of attempts to find a valid cluster.
              Default is 10, which means it will try to find a valid cluster up to 10 times.

        Returns:
            A single detected cluster motif.
        """
        if size is None:
            size = int(self.rng.integers(2, self.max_cluster_size + 1))

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
                        (existing_cluster + site).radius <= self.max_cluster_radius
                )
            ]

        def _detect_attempt(a):
            # Perform a single detection attempt.
            valid_indices = self._get_symbol_valid_indices(a)
            rand_idx = self.rng.choice(valid_indices)
            rand_indices = [rand_idx]
            c = ClusterMotif.from_atoms(a[[rand_idx]], indices=[rand_idx])
            for _ in range(size - 1):
                # Randomly select a site to grow the cluster around.
                neighbor_site_motifs = SiteDetector(
                    cutoff=self.cutoff,
                    symbols=self.symbols
                ).detect_around_site_indices(a, [rand_indices[-1]])
                deduplicated_site_motifs = _filter_neighbors(cluster, neighbor_site_motifs)
                if len(deduplicated_site_motifs) == 0:
                    return None  # Failed to grow the cluster.
                deduplicated_site_indices = [
                    int(site.indices[0]) for site in deduplicated_site_motifs
                ]
                rand_nn_idx = int(self.rng.integers(len(deduplicated_site_indices)))
                c += deduplicated_site_motifs[rand_nn_idx]
                rand_indices += [deduplicated_site_indices[rand_nn_idx]]
            return c

        for _ in range(n_attempts):
            cluster = _detect_attempt(atoms)
            if cluster is not None and len(cluster) == size:
                return cluster

        print(f"Warning: Failed to detect a cluster of size {size} in {n_attempts} attempts."
              f" Please check the structure and cutoff parameters.")

        return None
