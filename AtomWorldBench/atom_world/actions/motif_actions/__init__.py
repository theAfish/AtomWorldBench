from .base import BaseMotifAction
from ....common.registry import derived_class_factory

def motif_action_factory(
        action_name: str,
        *args,
        **kwargs):
    """Factory function to create an action instance based on the given type.

    Args:
        action_name (str): The type of action to create.
            Must be one of the alias names registered in the BaseAction class.
            See documentation of `common.registry.register` for more details.
        *args: Positional arguments to pass to the action's constructor.
        **kwargs: Keyword arguments to pass to the action's constructor.
    Returns:
        BaseMotifAction: An instance of the requested action type.
    """
    return derived_class_factory(action_name, BaseMotifAction, *args, **kwargs)
