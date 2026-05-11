"""Global verifier registry and chain utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from .base import BaseVerifier

# Default verifier chain — mirrors the existing simple/verbose task logic.
# Tasks that need different pipelines declare their own list in the dataset.
DEFAULT_VERIFIER_CHAIN: List[str] = [
    "output_format",
    "cif_parsing",
    "atom_count",
    "structure_match",
]


class VerifierRegistry:
    """Singleton registry mapping verifier names to instances.

    Verifiers self-register via the ``@VerifierRegistry.register(name)``
    class decorator.
    """

    _registry: Dict[str, "BaseVerifier"] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str):
        """Class decorator that instantiates and registers a verifier.

        Usage::

            @VerifierRegistry.register("my_verifier")
            class MyVerifier(BaseVerifier):
                ...
        """
        def decorator(verifier_cls):
            instance = verifier_cls()
            instance.name = name
            cls._registry[name] = instance
            return verifier_cls

        return decorator

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, name: str) -> "BaseVerifier":
        """Return the registered verifier instance for *name*.

        Raises ``ValueError`` for unknown names so callers get a clear
        message rather than a ``KeyError``.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown verifier {name!r}. "
                f"Available: {sorted(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def get_chain(cls, names: List[str]) -> List["BaseVerifier"]:
        """Return an ordered list of verifier instances for *names*."""
        return [cls.get(n) for n in names]

    @classmethod
    def available(cls) -> List[str]:
        """Return a sorted list of all registered verifier names."""
        return sorted(cls._registry.keys())
