from .base import BaseDescriptionStyle

from ...utils.class_utils import derived_class_factory, class_name_from_str


def description_style_factory(
        style_name: str,
        *args,
        **kwargs
):
    """Factory function to create a description style based on the given name.

    Args:
        style_name (str): The name of the description style to create.
            Must be the first part of the class name, e.g., "coord" for CoordDescriptionStyle.
        *args: Positional arguments to pass to the style constructor.
        **kwargs: Keyword arguments to pass to the style constructor.

    Returns:
        BaseDescriptionStyle: An instance of the requested description style.
    """
    class_name = class_name_from_str(style_name + "-description-style")
    return derived_class_factory(class_name, BaseDescriptionStyle, *args, **kwargs)
