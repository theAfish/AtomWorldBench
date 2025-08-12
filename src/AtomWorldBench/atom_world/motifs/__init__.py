from ...utils.class_utils import class_name_from_str, derived_class_factory

from ..motifs.site_collection_motifs.base import BaseSiteCollectionMotif


# TODO: polyhedra, etc.
# TODO: implement a region motif that can be used to include all sites in a region in space.
#  This will enable actions such as "move all atoms in a region towards or away from a point",
def motif_factory(
        motif_name: str,
        *args,
        **kwargs
):
    """Factory function to create a detector instance based on the provided name.

    Args:
        motif_name (str): The type of motif to create.
            Must be the first part of the class name, e.g., "cluster" for ClusterMotif.
        *args: Positional arguments to pass to the motif's constructor.
        **kwargs: Keyword arguments to pass to the motif's constructor.
    Returns:
        BaseSiteCollectionMotif: An instance of the specified motif type.
    """
    class_name = class_name_from_str(motif_name + "-motif")
    return derived_class_factory(class_name, BaseSiteCollectionMotif, *args, **kwargs)