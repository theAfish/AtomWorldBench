from .base import BaseAction
from ...utils.class_utils import derived_class_factory

from pymatgen.core import Structure

def action_factory(
        action_type: str,
        structure: Structure,
        **kwargs) -> BaseAction:
    """Factory function to create an action instance based on the action type.

    Args:
        action_type (str): The type of action to create.
        structure (pymatgen.core.structure.Structure): The structure to be modified by the action.
        kwargs: Additional keyword arguments to pass to the action constructor.

    Returns:
        BaseAction: An instance of a subclass of BaseAction corresponding to the action type.
    """
    return derived_class_factory(
        class_name=action_type,
        base_class=BaseAction,
        structure=structure,
        **kwargs
    )