from ...common.registry import derived_class_factory
from .base import BaseDetector

# TODO: replace detector classes with motif method instead.
def detector_factory(
        detector_name: str,
        *args,
        **kwargs
):
    """Factory function to create a detector instance based on the provided name.

    Args:
        detector_name (str): The type of detector to create.
            Must be one of the alias names registered in the `BaseDetector` class.
            See documentation of `common.registry.register` for more details.
        *args: Positional arguments to pass to the detector's constructor.
        **kwargs: Keyword arguments to pass to the detector's constructor.
    Returns:
        BaseDetector: An instance of the specified detector type.
    """
    return derived_class_factory(detector_name, BaseDetector, *args, **kwargs)