from .base import BaseAction
from ...utils.class_utils import class_name_from_str, derived_class_factory

def action_factory(
        action_type: str,
        *args,
        **kwargs):
    """Factory function to create an action instance based on the given type.
    Args:
        action_type (str): The type of action to create.
            Must be the first part of the class name, e.g., "move" for MoveAction.
        *args: Positional arguments to pass to the action's constructor.
        **kwargs: Keyword arguments to pass to the action's constructor.
    Returns:
        BaseAction: An instance of the requested action type.
    """
    class_name = class_name_from_str(action_type + "-action")
    return derived_class_factory(class_name, BaseAction, *args, **kwargs)