from ...utils.class_utils import derived_class_factory

from .base import BaseDetector

def detector_factory(
        detector_type: str,
        *args,
        **kwargs
) -> BaseDetector:
    """
    Factory function to create a detector instance based on the provided name.

    Args:
        detector_type (str): The type of detector to create.
        *args: Positional arguments to pass to the detector's constructor.
        **kwargs: Keyword arguments to pass to the detector's constructor.
    Returns:
        BaseDetector: An instance of the specified detector type.
    """
    return derived_class_factory(detector_type, BaseDetector, *args, **kwargs)