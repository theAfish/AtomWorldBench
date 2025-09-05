from ...common.registry import derived_class_factory

from .base import BaseMotif


# TODO: polyhedra, etc.
# TODO: Add a type of motif that allows selecting all atoms of a certain element.
def motif_factory(
        motif_name: str,
        *args,
        **kwargs
):
    """Factory function to create a detector instance based on the provided name.

    Args:
        motif_name (str): The type of motif to create.
            Must be one of the alias names registered in the BaseMotif class.
            See documentation of `common.registry.register` for more details.
        *args: Positional arguments to pass to the motif's constructor.
        **kwargs: Keyword arguments to pass to the motif's constructor.
    Returns:
        BaseSiteCollectionMotif: An instance of the specified motif type.
    """
    return derived_class_factory(motif_name, BaseMotif, *args, **kwargs)
