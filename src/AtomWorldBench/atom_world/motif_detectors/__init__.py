from ...utils.class_utils import derived_class_factory, class_name_from_str

from .base import BaseDetector

def detector_factory(
        detector_name: str,
        *args,
        **kwargs
):
    """
    Factory function to create a detector instance based on the provided name.

    Args:
        detector_name (str): The type of detector to create.
            Must be the first part of the class name, e.g., "site" for SiteDetector.
        *args: Positional arguments to pass to the detector's constructor.
        **kwargs: Keyword arguments to pass to the detector's constructor.
    Returns:
        BaseDetector: An instance of the specified detector type.
    """
    class_name = class_name_from_str(detector_name + "-detector")
    return derived_class_factory(class_name, BaseDetector, *args, **kwargs)