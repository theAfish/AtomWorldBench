"""Mixin classes to adjust the behavior of implemented subclasses."""

from __future__ import annotations

import inspect
import itertools
from abc import ABC
from typing import Any, Union, Callable, Sequence, Mapping


# ---- Type aliases -----------------------------------------------------------

# A "condition" attached to a parameter inside a mode definition:
# - None       -> the parameter just needs to be provided and not None
# - (fn, desc) -> fn(value) must be True; 'desc' is used in human-friendly errors
Condition = Union[tuple[Callable[[Any], bool], str], None]

# For one mode: mapping of parameter name -> Condition
ModeRule = Mapping[str, Condition]

# Full mode table for a class:
# - Special keys supported (see docs below):
#     "_excluded": Sequence[str]
#     "_combinations": Sequence[Mapping[str, Union[str, ModeRule]]]
#       - Each element in the list is a *block* describing a product space.
#       - In a block, each key except "name_template" is a *dimension*,
#         whose value is a mapping of *option_name* -> *ModeRule* for that dimension.
#       - Optional key "name_template" (str) formats the final mode name using
#         Python's str.format with each dimension name as a field.
# - Other top-level keys (that do not start with "_") are explicit, fully-specified modes.
ModeDefinitions = Mapping[
    str,
    Union[
        Sequence[str],                                   # For _excluded
        ModeRule,                                        # For explicit modes
        Sequence[Mapping[str, Union[str, ModeRule]]],    # For _combinations (list of blocks)
    ],
]


class MultiModeInitMixin(ABC):
    """A small framework for classes that support *multiple init modes* in their constructor.

    What it does:
    * inspects the keyword arguments passed to __init__
    * detects which "mode" those arguments describe (based on your `mode_definitions`)
    * validates each argument with simple "conditions"
    * formats/transforms argument values via `kwargs_formatting_functions`
    * exposes a read-only `mode_flag` telling you which mode was selected

    How to use in a subclass:
    1) Provide a class attribute `mode_definitions` (see below).
    2) (Optional) Provide a class attribute `kwargs_formatting_functions`.
    3) Call `super().__init__(**kwargs)` in your subclass __init__ *after* any essential
       pre-setup you need for error messages, but *before* you rely on formatted attributes.
       For suppressing IDE warnings, before calling `super().__init__`, you can
       set self.<param_name> to None for every argument you expect to appear later.
    4) (Optional) Override `__pre_init__` and/or `__post_init__` for extra checks.

    ---------------------------------------------------------------------------
    Friendly explanations
    ---------------------------------------------------------------------------

    • ModeDef / mode_definitions  — flat modes and combinational modes
      You define *modes* of construction for the class. Each mode is a dictionary
      describing which parameters are allowed and what each must satisfy.

      You can mix two styles:

      (A) Flat, explicit modes
          Structure (dictionary):
              {
                "_excluded": ["param_a", "param_b", ...],  # not used to decide mode
                "mode_name_1": {
                    "required_param": None,                      # must be provided and not None
                    "constrained_param": (lambda x: x > 0, "x > 0"),
                    ...
                },
                "mode_name_2": { ... },
                ...
              }
          Rules:
            - Every key except "_excluded" (and "_combinations", see below) is treated as a *mode name*.
            - For each (param -> Condition):
                * Condition is None      -> “param must be present and not None”
                * Condition = (fn, desc) -> `fn(value)` must be True; 'desc' appears in errors.
            - During detection:
                * All required params in a mode must be provided (not missing), not None, and satisfy their condition.
                * No *extra* non-excluded params are allowed for that mode.
            - Exactly one mode must match, otherwise an informative ValueError is raised.

      (B) Combinational modes (avoid M×N boilerplate)
          When a mode is conceptually a *product* of several dimensions (e.g. rotation type ×
          reference frame), you can describe the *dimensions* and their *options* once, and
          the mixin will automatically generate all valid combinations.

          Example structure:
              {
                "_excluded": ["name"],

                # 1) Explicit, standalone modes can still live at top-level keys (optional)
                "axis_rotation_relative_to_self": {
                    "axis": None,
                    "self_flag": None
                },

                # 2) A special "_combinations" list for product-style modes
                "_combinations": [ {
                    # Dimensions (each key except "name_template" is a dimension)
                    "rotation_type": {
                        "axis":   { "axis": None },
                        "matrix": { "matrix": None },
                    },
                    "relative_to": {
                        "coord": { "reference_coord": None },
                        "motif": { "reference_motif": None },
                    },

                    # Optional naming template for the final mode name
                    # The fields are dimension names; values are the chosen option names.
                    "name_template": "{rotation_type}_rotation_relative_to_{relative_to}"
                  } ]
              }

          This will auto-generate these modes (plus the explicit one above):
              "axis_rotation_relative_to_coord":  { "axis": None, "reference_coord": None }
              "axis_rotation_relative_to_motif":  { "axis": None, "reference_motif": None }
              "matrix_rotation_relative_to_coord":{ "matrix": None, "reference_coord": None }
              "matrix_rotation_relative_to_motif":{ "matrix": None, "reference_motif": None }

          Why this is useful:
            - You write each dimension’s options only once (no M×N duplication).
            - You can keep independent, non-product modes explicit at the top level
              (e.g., "axis_rotation_relative_to_self") so they do not combine with others.

          Notes:
            - If you don’t provide "name_template", a default name will be constructed by
              joining the chosen option names with underscores (e.g. "axis_coord").
            - Parameter name collisions across dimensions are rejected (to avoid ambiguity).
            - Explicitly named modes and auto-generated names cannot collide.

    • kwargs_formatting_functions
      Lets you transform/normalize values during initialization (e.g., type conversions,
      canonicalization, per-mode tweaks).

      Structure (dictionary):
          {
            "param_name": callable,
            ...
          }
      Standard signature for a formatter callable:
          def format_param(value, *, mode_flag: Optional[str] = None) -> Any
      If your callable accepts the keyword-only argument `mode_flag`, it will be passed.
      Otherwise, it will be called as `callable(value)`.
      This callable must return an object, not simply true/false or None.

    Attribute behavior
    ------------------
    - After successful detection/formatting, each kwarg becomes an instance attribute with the
      (possibly) formatted value: `self.<param_name> = <formatted_value>`.
    - `self.mode_flag` is read-only (immutable after construction).

    Safety & ergonomics
    -------------------
    - `__init_subclass__` validates that your subclass provides a well-formed `mode_definitions`.
    - The mode detection distinguishes “parameter not provided” vs “provided as None”.
    - Excluded parameters (listed in `_excluded`) never participate in mode detection,
      but are still allowed as inputs and will be set/formatted like others.
    - Combinational modes live under the `_combinations` list and are flattened
      once at class definition time; explicit modes remain as-is.

    Example
    -------
    class AddCircle(MultiModeInitMixin):
        # Two distinct ways to construct:
        #   - "by_radius": center + radius
        #   - "by_points": three distinct points
        mode_definitions = {
            "_excluded": ["metadata"],
            "by_radius": {
                "center": (lambda c: isinstance(c, (tuple, list)) and len(c) == 2, "2D point"),
                "radius": (lambda r: r > 0, "radius > 0"),
            },
            "by_points": {
                "p1": (lambda p: isinstance(p, (tuple, list)) and len(p) == 2, "2D point"),
                "p2": (lambda p: isinstance(p, (tuple, list)) and len(p) == 2, "2D point"),
                "p3": (lambda p: isinstance(p, (tuple, list)) and len(p) == 2, "2D point"),
            },
        }

        # Normalize inputs (lowercase metadata; ensure tuples; etc.)
        kwargs_formatting_functions = {
            "metadata": lambda v, *, mode_flag=None: (v or "").lower(),
            "center":   lambda v, *, mode_flag=None: tuple(v) if v is not None else None,
            "p1":       lambda v, *, mode_flag=None: tuple(v) if v is not None else None,
            "p2":       lambda v, *, mode_flag=None: tuple(v) if v is not None else None,
            "p3":       lambda v, *, mode_flag=None: tuple(v) if v is not None else None,
        }

        def __init__(self, **kwargs):
            # you may prepare defaults or sanity checks here if needed
            super().__init__(**kwargs)  # detects mode, formats, and sets attributes
            # now self.mode_flag is available, along with formatted attributes

        def __post_init__(self) -> None:
            if self.mode_flag == "by_points":
                if len({self.p1, self.p2, self.p3}) < 3:
                    raise ValueError("Points must be distinct in 'by_points' mode.")

    """

    # Subclasses override these two:
    kwargs_formatting_functions: Mapping[str, Callable[..., Any]] = {}
    # Avoid providing an implicit empty mode like {"default": {}}
    mode_definitions: ModeDefinitions = {"_excluded": []}

    # Internal cache of the flattened, validation-ready mode table for this class.
    # This is computed at class definition time in __init_subclass__.
    _flattened_mode_definitions: Mapping[str, Union[Sequence[str], Mapping[str, Condition]]] = {}

    # ---- Class-level validation & flattening (runs when a subclass is defined) ------------
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        md = getattr(cls, "mode_definitions", None)
        if not isinstance(md, Mapping):
            raise TypeError(f"{cls.__name__}.mode_definitions must be a mapping.")

        if "_excluded" not in md:
            raise TypeError(f"{cls.__name__}.mode_definitions must include the '_excluded' key.")

        excluded = md.get("_excluded")
        if not isinstance(excluded, Sequence):
            raise TypeError(f"{cls.__name__}._excluded must be a sequence (e.g. list or tuple).")

        # Validate explicit modes (top-level keys that do not start with "_")
        explicit_modes: dict[str, dict[str, Condition]] = {}
        for mode_name, rule in md.items():
            if mode_name.startswith("_"):
                continue
            if not isinstance(rule, Mapping):
                raise TypeError(
                    f"{cls.__name__}.mode_definitions['{mode_name}'] must be a mapping of param -> Condition."
                )
            normalized: dict[str, Condition] = {}
            for param_name, cond in rule.items():
                if param_name in excluded:
                    raise ValueError(
                        f"{cls.__name__}: parameter '{param_name}' is both in mode '{mode_name}' and in _excluded."
                    )
                if cond is None:
                    normalized[param_name] = None
                    continue
                if (
                    not isinstance(cond, tuple)
                    or len(cond) != 2
                    or not callable(cond[0])
                    or not isinstance(cond[1], str)
                ):
                    raise TypeError(
                        f"{cls.__name__}.{mode_name}.{param_name} must be None or (callable, 'description')."
                    )
                normalized[param_name] = cond
            explicit_modes[mode_name] = normalized

        # Validate and flatten combinational modes (optional)
        flattened_from_combos: dict[str, dict[str, Condition]] = {}
        if "_combinations" in md:
            all_blocks = md["_combinations"]
            if not isinstance(all_blocks, Sequence):
                raise TypeError(f"{cls.__name__}._combinations must be a sequence of blocks.")

            for combos_id, block in enumerate(all_blocks):
                if not isinstance(block, Mapping):
                    raise TypeError(f"{cls.__name__}._combinations[{combos_id}] must be a mapping.")

                name_template = block.get("name_template", None)
                if name_template is not None and not isinstance(name_template, str):
                    raise TypeError(
                        f"{cls.__name__}._combinations[{combos_id}].name_template must be a str if provided."
                    )

                # Each key in block except "name_template" is a dimension
                dimensions = [(dim, opts) for dim, opts in block.items() if dim != "name_template"]
                if not dimensions:
                    raise ValueError(
                        f"{cls.__name__}._combinations[{combos_id}] must define at least one dimension."
                    )

                # Validate per-dimension options
                dim_names: list[str] = []
                dim_options_list: list[list[tuple[str, dict[str, Condition]]]] = []
                for dim_name, options in dimensions:
                    if not isinstance(options, Mapping) or not options:
                        raise TypeError(
                            f"{cls.__name__}._combinations[{combos_id}]['{dim_name}'] must be a "
                            f"non-empty mapping of option -> ModeRule."
                        )
                    dim_names.append(dim_name)
                    # Normalize each option's rule like explicit modes
                    normalized_opts: list[tuple[str, dict[str, Condition]]] = []
                    for opt_name, opt_rule in options.items():
                        if not isinstance(opt_rule, Mapping):
                            raise TypeError(
                                f"{cls.__name__}._combinations[{combos_id}]['{dim_name}']['{opt_name}'] "
                                f"must be a mapping of param -> Condition."
                            )
                        norm_rule: dict[str, Condition] = {}
                        for param_name, cond in opt_rule.items():
                            if param_name in excluded:
                                raise ValueError(
                                    f"{cls.__name__}: param '{param_name}' is both in _excluded and used in "
                                    f"_combinations[{combos_id}] dim '{dim_name}' option '{opt_name}'."
                                )
                            if cond is None:
                                norm_rule[param_name] = None
                            else:
                                if (
                                    not isinstance(cond, tuple)
                                    or len(cond) != 2
                                    or not callable(cond[0])
                                    or not isinstance(cond[1], str)
                                ):
                                    raise TypeError(
                                        f"{cls.__name__}._combinations[{combos_id}]['{dim_name}']['{opt_name}']."
                                        f"{param_name} must be None or (callable, 'description')."
                                    )
                                norm_rule[param_name] = cond
                        normalized_opts.append((opt_name, norm_rule))
                    dim_options_list.append(normalized_opts)

                # Build Cartesian product of options across all dimensions
                for combo in itertools.product(*dim_options_list):
                    # combo is a tuple of (opt_name, rule) per dimension, in dim_names order
                    option_names = [opt_name for (opt_name, _rule) in combo]

                    # Construct mode name
                    if name_template:
                        fmt_kwargs = {dim: opt for dim, opt in zip(dim_names, option_names)}
                        mode_name = name_template.format(**fmt_kwargs)
                    else:
                        mode_name = "_".join(option_names)

                    # Merge rules (check for parameter collisions)
                    merged: dict[str, Condition] = {}
                    for (_opt_name, rule_dict) in combo:
                        for param, cond in rule_dict.items():
                            if param in merged:
                                raise ValueError(
                                    f"{cls.__name__}: parameter '{param}' appears in multiple parts of "
                                    f"_combinations[{combos_id}] combo mode '{mode_name}'."
                                )
                            merged[param] = cond

                    if mode_name in explicit_modes or mode_name in flattened_from_combos:
                        raise ValueError(
                            f"{cls.__name__}: duplicate mode name '{mode_name}' from "
                            f"_combinations[{combos_id}] clashes with an existing mode."
                        )
                    flattened_from_combos[mode_name] = merged

        # Build the final flattened table (explicit + combos) with the shared _excluded
        cls._flattened_mode_definitions = {"_excluded": list(excluded)}
        cls._flattened_mode_definitions.update(explicit_modes)
        cls._flattened_mode_definitions.update(flattened_from_combos)

    # ---- Make mode_flag immutable after initialization -----------------------
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_mode_flag" and getattr(self, "_mode_flag", None) is not None:
            raise AttributeError("mode_flag is immutable after initialization.")
        super().__setattr__(name, value)

    # ---- Public API ----------------------------------------------------------
    @property
    def mode_flag(self) -> str:
        """The detected mode name (read-only)."""
        return self._mode_flag

    # ---- Lifecycle -----------------------------------------------------------
    def __init__(self, **kwargs: Any):
        """
        Constructor that:
        1) runs __pre_init__(kwargs)
        2) detects and stores mode_flag
        3) validates & formats kwargs
        4) runs __post_init__()
        """
        self.__pre_init__(kwargs)
        self._mode_flag = self._detect_mode(kwargs)  # frozen afterwards
        self._set_and_format_kwargs(kwargs)
        self.__post_init__()

    def __pre_init__(self, kwargs: dict[str, Any]) -> None:
        """Hook for subclasses to do pre-initialization checks."""
        pass

    def __post_init__(self) -> None:
        """Hook for subclasses to do post-initialization checks."""
        pass

    # ---- Core helpers --------------------------------------------------------
    def _detect_mode(self, kwargs: dict[str, Any]) -> str:
        """
        Decide which mode matches the provided kwargs.

        A mode matches if:
        - every required param in that mode is PRESENT (key exists), NOT None, and satisfies its condition
        - there are no extra non-excluded params present (present and not None)

        Exactly one mode must match.
        """
        md = self._flattened_mode_definitions or self.mode_definitions
        excluded = set(md.get("_excluded", []))

        candidates: list[str] = []
        failures: dict[str, list[str]] = {}

        for mode_name, rule in md.items():
            if mode_name == "_excluded":
                continue

            # mypy/pyright friendly: rule is Mapping[str, Condition]
            assert isinstance(rule, Mapping)
            failures[mode_name] = []

            # Check required params
            required_params = set(rule.keys())
            for pname, cond in rule.items():
                if pname not in kwargs:
                    failures[mode_name].append(f"{pname} is required but not provided.")
                    continue

                val = kwargs[pname]
                if val is None:
                    failures[mode_name].append(f"{pname} is provided as None.")
                    continue

                if cond is not None:
                    fn, desc = cond
                    try:
                        ok = bool(fn(val))
                    except Exception as e:
                        failures[mode_name].append(f"{pname} condition [{desc}] raised {e!r}.")
                        ok = False
                    if not ok:
                        failures[mode_name].append(f"{pname} fails condition [{desc}].")

            # Check for extra params (present & not None) that are not in this mode nor excluded
            allowed = required_params | excluded
            extras = [k for k in kwargs.keys() if k not in allowed and kwargs[k] is not None]
            if extras:
                failures[mode_name].append(f"extra params not allowed in '{mode_name}': {extras}")

            if len(failures[mode_name]) == 0:
                candidates.append(mode_name)

        if len(candidates) == 0:
            raise ValueError(
                f"No mode detected for {self.__class__.__name__}. "
                f"kwargs={kwargs}, failures={failures}."
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous modes for {self.__class__.__name__}. Possible modes: {candidates}"
            )
        return candidates[0]

    def _set_and_format_kwargs(self, kwargs: dict[str, Any]) -> None:
        """
        For each kwarg, set it as an attribute. If a formatter is registered in
        `kwargs_formatting_functions`, apply it first.

        Formatter signature options:
          def f(value) -> Any
          def f(value, *, mode_flag: Optional[str] = None) -> Any
        """
        for key, value in kwargs.items():
            fn = self.kwargs_formatting_functions.get(key)
            if fn is None:
                setattr(self, key, value)
                continue

            try:
                sig = inspect.signature(fn)
                # Reserve mode_flag as a tunable keyword.
                if "mode_flag" in sig.parameters:
                    formatted = fn(value, mode_flag=self.mode_flag)
                else:
                    formatted = fn(value)
            except TypeError as e:
                raise TypeError(
                    f"Formatter for '{key}' must be callable(value) or "
                    f"callable(value, *, mode_flag: Optional[str] = None);"
                    f" got {fn!r}. Error message: {e}."
                )

            setattr(self, key, formatted)
