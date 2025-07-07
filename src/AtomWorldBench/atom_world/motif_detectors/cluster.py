from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike
from pymatgen.core import Structure
from pymatgen.util.typing import SpeciesLike

from .base import BaseDetector
from ..motifs.cluster import ClusterMotif


def grow_cluster(
        structure: Structure,
        root_cluster: List[int],
        available_site_ids: List[int],
        max_cluster_radius: float,
) -> List[List[int]]:
    """Grow a cluster by adding available sites to the root cluster.

    Args:
        structure (Structure): The structure containing the sites.
        root_cluster (List[int]): The initial cluster to grow from, containing indices of sites.
        available_site_ids (List[int]): Indices of available sites to consider for growth.
        max_cluster_radius (float): Maximum allowed radius for the cluster.

    Returns:
        List[List[int]]: A list containing the indices of the sites in the grown clusters.
    """
    for ii in available_site_ids:
        site = structure[ii]
        attempted


class ClusterDetector(BaseDetector):
    """Detects clusters of atoms in a structure.

    This detector identifies clusters based on the proximity of atoms and their species.
    It can be used to find groups of atoms that are close together and share similar properties.
    """
    def __init__(
            self,
            radius: float = 3.0,
            max_cluster_size: int = 2,
            max_cluster_radius: Optional[float] = None,
            must_include_center: bool = True,
            species_to_include: Optional[List[SpeciesLike]] = None,
    ):
        """Initialize the ClusterDetector with a default radius.

        Clusters will only be returned if they meet the specified criteria:
         1, Fewer sites than the specified `cluster_size`
         2, Radius less than or equal to `max_cluster_radius`.
         3, If `must_include_center` is True, the center atom must be part of the cluster.
         4, If `species_to_include` is specified, the cluster must contain only the specified species.

        Args:
            radius (float): The default radius for detecting clusters around fractional coordinates.
             Default is 3.0 Angstroms.
            max_cluster_size (int):
             Maximum number of atoms in each detected cluster. Default is 2 (doublet).
            max_cluster_radius (Optional[float]):
             Maximum allowed cluster radius for detection.
             Clusters larger than this radius will not be detected.
             Default is None, will be set to `radius` if not provided.
            must_include_center (bool):
             If True, the center atom must be included in the detected cluster. Default is True.
            species_to_include (Optional[List[SpeciesLike]]):
             List of species that must be included in the detected clusters. If None,
             all species are considered. Default is None.
        """
        self.radius = radius
        self.max_cluster_size = max_cluster_size
        self.max_cluster_radius = max_cluster_radius if max_cluster_radius is not None else radius
        self.must_include_center = must_include_center
        self.species_to_include = species_to_include

    def detect_around_frac_coords(
            self,
            structure: Structure,
            frac_coords: ArrayLike,
    ) -> List[ClusterMotif]:
        """Detect clusters in the given structure around fractional coordinates.

        First detects all atoms within the specified radius from the given fractional coordinates,
        then groups them into clusters based on their proximity.

        Args:
            structure (Structure): The structure to analyze.
            frac_coords (ArrayLike): Fractional coordinates for the cluster detection center.
                Must be a one-dimensional array of shape (3,).

        Returns:
            List of detected clusters.
        """
        # Convert fractional coordinates to Cartesian coordinates
        cart_coords = structure.lattice.get_cartesian_coords(frac_coords)
        neighbors = structure.get_sites_in_sphere(cart_coords, self.radius)

        # Filter neighbors based on species.
        if self.species_to_include is not None:
            neighbors = [site for site in neighbors if site.specie in self.species_to_include]

        if self.must_include_center:
            center = neighbors[np.argmin(site.nn_distance for site in neighbors)]
            indices_in_clusters = [[center.index]]
        else:
            indices_in_clusters = [[site.index for site in neighbors]]
