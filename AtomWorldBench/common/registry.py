"""Implement subclass registry and factory methods.

Third-party code should use load_plugins() method in their __init__.py to register
all their subclasses automatically.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable, Union, Optional
import importlib
import pkgutil
from types import ModuleType
import inspect
import logging


# Not to be modified explicitly.
_REGISTRY: dict[type, dict[str, type]] = {}


def aliases_from_class_name(class_name: str) -> set[str]:
    """Generate possible aliases from class names.

    Args:
        class_name (str): The name of the class to generate aliases for.
            All class must be named in CamelCase format for AtomWorldBench to work properly,
             e.g., "MyClass". (with the first letter capitalized and no underscores or hyphens).

    Returns:

        set[str]: A set of aliases derived from the class name.
    """
    words = re.findall(r"[A-Z][a-z0-9]*", class_name) or [class_name]
    kebab = "-".join(w.lower() for w in words)
    snake = "_".join(w.lower() for w in words)
    lower = "".join(w.lower() for w in words)
    return {class_name, kebab, snake, lower}


def register(base_cls: type, aliases: Optional[list[str]]=None, overwrite: bool = False):
    """Register a concrete subclass under a set of aliases for an abstract base class.

    The decorated class must not be abstract and must inherit from the specified base class.
    Args:
        base_cls (type): The abstract base class to register
            the subclass under.
        aliases (Optional[list[str]]): A list of aliases to register the class under.
            If not provided, aliases will be generated from the class name.
        overwrite (bool): If True, allow overwriting existing aliases.
            If False, raise an error if any alias is already registered. Default is False.
    Returns:
        A decorator that registers the class under the specified aliases.
    """
    # enforce: base must be abstract, subclass must be concrete (your new rules)
    if not inspect.isabstract(base_cls):
        raise TypeError(f"{base_cls.__name__} must be an abstract class for registration.")

    def decorator(cls):
        # Must inherit base
        if not issubclass(cls, base_cls):
            raise TypeError(f"{cls.__name__} must inherit {base_cls.__name__} to register.")

        # Must be concrete (no abstract methods left)
        if inspect.isabstract(cls):
            raise TypeError(f"{cls.__name__} cannot be abstract when registered.")

        # Get value if value already present, else set key's value to default empty dict and
        # return that empty dict.
        reg = _REGISTRY.setdefault(base_cls, {})

        # Build all aliases (class-name variants + user-specified)
        alias_set = set(aliases or [])
        alias_set |= aliases_from_class_name(cls.__name__)

        # Normalize (optional but recommended)
        normalized = {a.strip() for a in alias_set if a is not None and a.strip()}

        # Detect collisions
        collisions = [a for a in normalized if a in reg]
        if collisions and not overwrite:
            # If any alias already taken, error out with the first conflict for clarity
            taken = collisions[0]
            existing = reg[taken]
            raise ValueError(
                f"Alias '{taken}' already registered to {existing.__name__} for base {base_cls.__name__}. "
                f"Refusing to register {cls.__name__}. "
                f"(Pass overwrite=True if you intend to replace it.)"
            )

        # Apply mappings (overwriting if requested)
        for a in normalized:
            reg[a] = cls

        return cls

    return decorator


def get_registered(base: type) -> dict[str, type]:
    """Get all registered subclasses for a given base class.

    Returns a shallow copy of the registry for the base class to
    prevent external modifications.
    """
    # Create a shallow copy to prevent external modifications.
    return dict(_REGISTRY.get(base, {}))


def derived_class_factory(class_name: str, base_class: type, *args, **kwargs):
    """Factory function to create an instance of a derived class.

    Args:
        class_name (str): The name of the class to instantiate.
            Allowed class name formats are:
            - CamelCase (e.g., "MyClass")
            - kebab-case (e.g., "my-class")
            - snake_case (e.g., "my_class")
            - lower case (e.g., "myclass")
            - any other alias as registered in subclass definition.
        base_class (type): The base class that the derived class must inherit from.
        *args: Positional arguments for the class constructor
        **kwargs: Keyword arguments for the class constructor
    Returns:
        object: An instance of the derived class with the given arguments.
    """
    sub_classes = get_registered(base_class)

    if class_name not in sub_classes:
        suggestions = ", ".join(sorted(sub_classes.keys()))
        raise NotImplementedError(
            f"{class_name} is not implemented."
            f" Available classes: {suggestions}."
        )

    cls = sub_classes[class_name]

    return cls(*args, **kwargs)


def load_plugins(
        package: Union[str, ModuleType],
        *,
        include: Optional[Callable[[str], bool]] = None,
        exclude: Optional[Callable[[str], bool]] = None,
        strict: bool = False,
        before_import: Optional[Callable[[str], None]] = None,
        after_import: Optional[Callable[[str, ModuleType], None]] = None,
        logger: Optional[logging.Logger] = None,
) -> list[str]:
    """Load all plugins from a package.

    This will import all modules in the package and register them.
    Used for third-party code to register their subclasses automatically.

    Args:
        package (Union[str, ModuleType]): The package to load plugins from.
        include (Optional[Callable[[str], bool]]): Filter for module names to include.
        exclude (Optional[Callable[[str], bool]]): Filter for module names to exclude.
        strict (bool): If True, raise errors. If False, log errors and continue.
        before_import (Optional[Callable[[str], None]]): Hook called before importing.
        after_import (Optional[Callable[[str, ModuleType], None]]): Hook called after importing.
        logger (Optional[logging.Logger]): Logger to use. If None, uses default logger.

    Returns:
        list[str]: A list of names of successfully loaded modules.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if isinstance(package, str):
        package = importlib.import_module(package)

    if not hasattr(package, "__path__"):
        return [package.__name__]

    loaded: list[str] = []
    prefix = package.__name__ + "."

    for modinfo in pkgutil.walk_packages(package.__path__, prefix=prefix):
        name = modinfo.name
        if include and not include(name):
            continue
        if exclude and exclude(name):
            continue
        try:
            if before_import:
                before_import(name)
            mod = importlib.import_module(name)
            loaded.append(name)
            if after_import:
                after_import(name, mod)
        except Exception as e:
            msg = f"[load_plugins] Failed to import {name}: {e}"
            if strict:
                raise RuntimeError(msg) from e
            else:
                logger.error(msg)
    return loaded
