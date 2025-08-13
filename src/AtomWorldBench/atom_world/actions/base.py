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

    See documentation for AtomWorldBench.mixin_classes for more details on how to
     implement actions, if it has multiple modes of initialization and operation.
    """

    def __init__(
            self,
            **kwargs
    ):
        """Initialize the BaseAction with an optional relative motif.

        Check specific subclass for definition of parameters.
        """
        # Pre initialization checks.
        self.__pre_init__(kwargs)
        # Check the mode if an action supports multiple modes.
        self._mode_flag = self._check_mode(kwargs)  # Protected from resetting.
        # Validate and format the kwargs using the defined formatting functions.
        self._set_and_format_kwargs(kwargs)
        # Post init modifications.
        self.__post_init__()

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
                        f"{param_name} required but not provided or only given None."
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
        """Set and format the kwargs using the defined formatting functions."""
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
        # No longer sets default relative_style, user need to explicitly specify
        # it whenever used in operation modes.

    def __pre_init__(self, kwargs):
        """Pre-initialization checks for the action.

        Can be overridden by subclasses to implement specific checks if necessary.
        By default, only checks for compatibility of relative_style.
        """
        if not self.mode_definitions:
            raise ValueError(
                f"Action {self.__class__.__name__} must define mode_definitions class attribute."
            )
        if not isinstance(self.mode_definitions, dict):
            raise TypeError(
                f"Action {self.__class__.__name__} mode_definitions must be a dictionary."
            )
        _check_relative_style_compatibility(
            kwargs.get("relative_style", None),
            kwargs.get("relative_to_motif", None),
        )

    def __post_init__(self):
        """Check if the inputs are compatible with the action.

        Can be overridden by subclasses to implement specific checks if necessary.
        """
        # Relative style compatibility check moved to _check_mode.
        # Implement other check here.
        pass

    def execute(self, atoms: Atoms, operated_motif: BaseMotif) -> Atoms:
        """Execute the action on the structure to generate the ground truth structure."""
        passed, message = self.check_compatibility(atoms, operated_motif)
        if passed:
            return self._execute(atoms, operated_motif)
        raise ValueError(
            f"Action {self.__class__.__name__} cannot be performed with the given"
            f" Atoms and motif. Reason: [{message}]."
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
        relative_to_motif = getattr(self, "relative_to_motif", None)
        if relative_to_motif is not None:
            # Check if the relative motif is in the structure.
            indices, message = relative_to_motif.find_indices_in_atoms(
                atoms,
                modify_indices_in_place=True
            )
            if indices is None:
                return False, f"reference motif not found in structure: {message}"
        if not np.allclose(
                atoms.cell.complete().array,
                motif.cell.complete().array,
                atol=1e-6
        ) or not np.all(
            atoms.pbc == motif.pbc
        ):
            return False, "the operated motif's cell/pbc does not match the atoms cell/pbc."
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
