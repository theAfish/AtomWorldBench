"""Defines the base class for actions in the atom world."""
from abc import ABC, abstractmethod
from typing import Tuple
from collections import defaultdict

from ase import Atoms
import numpy as np

from ..motifs.base import BaseMotif


class BaseAction(ABC):
    """Base class for actions that can be performed on crystal structures.

    An action is defined as an interface to change a pymatgen.structure object,
    yielding a new pymatgen structure object as the ground truth, as well as a
    prompt to large language models (LLMs) describing operations done to the structure.

    If you implement a new action, you can simply edit:
        1, `allowed_relative_styles` class attribute;
        2, `kwargs_and_formating_functions` class attribute, to specify how to check and
            format keyword arguments of __init__;
            This is a dictionary where keys are the names of the keyword arguments,
            and values are functions that take the value of the keyword argument to check.
        3, Call super().__init__ from your __init__ method to ensure proper checking
            and formatting of your keyword arguments.
            As all attributes will be dynamically assigned in `_set_and_format_kwargs` and
            will not be explicitly defined in the __init__ method, you should
            also remember to pre-assign None to every attribute used in your own codes,
            in order to declare all attributes for passing IDE static check.
        4, Override the `mode_definitions` class attribute to define the modes of the action.
            It is a dictionary where keys are mode flags and values are dictionaries
            with the allowed input parameter names in the mode as keys and the conditions they must
            satisfy in the mode as values. A condition is a tuple containing one checking function
            and one description, or None. None means the no extra condition, just require the
            parameter not to be None.
            The reserved key `_excluded` is a list used to specify parameters that are not used to
            check mode. Cannot overlap with any other keys in the mode definitions. They should be
            common parameters used by all modes, such as `position_fractional`.
            If required parameters are not provided, or additional parameters other than in the
            definition are provided, will fail to detect the corresponding mode.
            Read the `_check_mode` method for more details on how it works, and AddMotifAction's
            `mode_definitions` for an example of how to define modes.
        5, Inherit or override `_post_init_checks` method to implement any additional checks.
        6, Override the `_check_compatibility` method to implement the compatibility
            check to the operated motif.
        7, Override the `_execute` method to implement the action logic.
        8, Implement the `describe` method to generate a description of the action.
    """
    # The styles allowed to use the relative motif as action reference.
    allowed_relative_styles = []
    # A dictionary of functions to format kwargs for the action.
    # If a specific key is not found here, the value will be put directly into attribute.
    # Used by _set_and_format_kwargs method.
    kwargs_and_formating_functions = {}
    # Definition of modes. Used by _check_mode method to detect the mode of the action.
    # Must be overridden in all subclasses, otherwise will raise an error.
    mode_definitions = {}

    def __init__(
            self,
            **kwargs
    ):
        """Initialize the BaseAction with an optional relative motif.

        Check specific subclass for definition of parameters.
        """
        # Check the mode if an action supports multiple modes.
        self._mode_flag = self._check_mode(kwargs)  # Protected from resetting.
        # Validate and format the kwargs using the defined formatting functions.
        self._set_and_format_kwargs(kwargs)
        # Check if the relative motif and style are compatible.
        self._post_init_checks()

    @property
    def mode_flag(self) -> str:
        """Get the flag of mode of the action.

        Fixed at initialization, and can be used to check the mode of the action.
        Do not reset.
        """
        return self._mode_flag

    def _check_mode(self, kwargs):
        """Check the mode of the action from its init arguments.

        Do not modify this method in subclass unless absolutely necessary.
        Change the `mode_definitions` class attribute instead.
        """
        # A dictionary to accumulate failure messages during checking.
        mode_checking_results = defaultdict(list)
        excluded_params = self.mode_definitions.get("_excluded", [])
        # All other keys than `excluded` are considered as mode flags.
        mode_definitions = {
            k: v for k, v in self.mode_definitions.items()
            if k != "_excluded"
        }
        for mode_flag, mode_definition in mode_definitions.items():
            for param_name, condition in mode_definition.items():
                param = kwargs.get(param_name, None)
                if param is not None:
                    if condition is not None:
                        cond_func, desc = condition
                        if not cond_func(param):
                            mode_checking_results[mode_flag].append(
                                f"{param_name} provided but failed to meet"
                                f" the condition [{desc}] required by mode."
                            )
                else:
                    mode_checking_results[mode_flag].append(
                        f"{param_name} required but not provided."
                    )
            for param_name, param in kwargs.items():
                if param_name in excluded_params:
                    # Skip the excluded parameters.
                    continue
                if param_name not in mode_definition and param is not None:
                    mode_checking_results[mode_flag].append(
                        f"{param_name} provided but not allowed in the mode."
                    )
        detected_mode_flags = [k for k, v in mode_checking_results.items() if len(v) == 0]
        if len(detected_mode_flags) == 0:
            raise ValueError(
                f"No mode detected for the action. Explanation: {mode_checking_results}."
            )
        if len(detected_mode_flags) > 1:
            raise ValueError(
                f"Multiple modes detected for the action: {detected_mode_flags}. "
            )
        return detected_mode_flags[0]

    def _set_and_format_kwargs(self, kwargs):
        """Set and format the kwargs using the defined formatting functions.

        The relative_style attribute is always set to the first in
        allowed_relative_styles if None or not provided.
        """
        for key, value in kwargs.items():
            func = self.kwargs_and_formating_functions.get(key, None)
            if func is not None:
                func_input_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                if "mode_flag" in func_input_names:
                    # Mode-specific formatting and checking function.
                    setattr(self, key, func(value, mode_flag=self.mode_flag))
                else:
                    # Not mode-specific.
                    setattr(self, key, func(value))
            else:
                setattr(self, key, value)
        if (
                getattr(self, "relative_to_motif") is not None
                and getattr(self, "relative_style", None) is None
        ):
            print("Warning: relative_to_motif provided, but relative_style is not set."
                  " Setting relative_style to the first allowed style if any.")
            if self.allowed_relative_styles and len(self.allowed_relative_styles) > 0:
                self.relative_style = self.allowed_relative_styles[0]
            else:
                self.relative_style = None

    def _post_init_checks(self):
        """Check if the inputs are compatible with the action.

        Can be overridden by subclasses to implement specific checks if necessary.
        By default, it only checks the compatibility of relative motif and relative
        type.
        """
        self._check_relative_style_init_compatibility()

    def _check_relative_style_init_compatibility(self):
        """Check if the relative style is compatible with the given relative motif and action."""
        relative_style = getattr(self, 'relative_style', None)
        relative_to_motif = getattr(self, 'relative_to_motif', None)
        if relative_style is not None and relative_style not in self.allowed_relative_styles:
            raise ValueError(
                f"Relative style {relative_style} is not allowed. "
                f"Allowed styles are: {self.allowed_relative_styles}"
            )
        if relative_to_motif is not None:
            if relative_style not in relative_to_motif.allowed_relative_styles:
                raise ValueError(
                    f"Relative style {relative_style} is not allowed for the "
                    f"motif {relative_to_motif.__class__.__name__}. "
                    f"Allowed styles are: {relative_to_motif.allowed_relative_styles}"
                )
            elif relative_to_motif.allowed_relative_styles[relative_style] is not None:
                condition, failure_message = relative_to_motif.allowed_relative_styles[relative_style]
                if not condition(relative_to_motif):
                    raise ValueError(
                        f"Relative style {relative_style} is not compatible with the "
                        f"motif {relative_to_motif.__class__.__name__}. Reason: {failure_message}."
                    )

    def execute(self, atoms: Atoms, operated_motif: BaseMotif) -> Atoms:
        """Execute the action on the structure to generate the ground truth structure."""
        passed, message = self.check_compatibility(atoms, operated_motif)
        if passed:
            return self._execute(atoms, operated_motif)
        raise ValueError(
            f"Action {self.__class__.__name__} cannot be performed with the given"
            f" Atoms and motif. Reason: {message}."
        )

    @abstractmethod
    def _execute(self, atoms: Atoms, operated_motif: BaseMotif) -> Atoms:
        """Execute the action on the structure to generate the ground truth structure.

        Must be overridden by subclasses to implement specific actions.
        """
        pass

    @classmethod
    def class_compatibility(cls, motif: BaseMotif) -> bool:
        """Check if the action is compatible with the given Atoms and motif object.

        Args:
            motif: An instance of BaseMotif to check compatibility with.
        Returns:
            bool: True if the action is compatible with the motif, False otherwise.
        """
        return cls.__name__ in motif.allowed_actions

    def check_compatibility(self, atoms: Atoms, motif: BaseMotif) -> Tuple[bool, str]:
        """Check if the action is compatible with the given Atoms object.

        Args:
            atoms(Atoms): An instance of Atoms to check compatibility with.
            motif(BaseMotif): An instance of BaseMotif to check compatibility with.
        Returns:
            Tuple[bool, str]:
              True if the action is compatible with the Atoms and motif, False otherwise.
        """
        if not self.__class__.class_compatibility(motif):
            return False, "motif does not allow this action"
        if getattr(self, "relative_to_motif", None) is not None:
            # Check if the relative motif is in the structure.
            indices = getattr(self, "relative_to_motif", None).find_indices_in_atoms(
                atoms,
                modify_indices_in_place=True
            )
            if indices is not None:
                return True, ""
            return False, "relative_to_motif not found in the structure."
        if not np.allclose(
                atoms.cell.complete().array,
                motif.cell.complete().array,
                atol=1e-6
        ) or not np.all(
            atoms.pbc == motif.pbc
        ):
            return False, "The operated motif's cell/pbc does not match the atoms cell/pbc."
        return self._check_compatibility(atoms, motif)

    @abstractmethod
    def _check_compatibility(self, atoms: Atoms, motif: BaseMotif) -> Tuple[bool, str]:
        """Check if the action is compatible with the given Atoms object.

        Must be overridden by subclasses to implement specific compatibility checks.
        Args:
            atoms(Atoms): An instance of Atoms to check compatibility with.
            motif(BaseMotif): An instance of BaseMotif to check compatibility with.
        Returns:
            Tuple[bool,str]:
            True if the action is compatible with the Atoms and motif, False otherwise.
        """
        pass

    @abstractmethod
    def describe(self, motif: BaseMotif, **kwargs) -> str:
        """Generate a description of the action to be performed on the structure."""
        pass
