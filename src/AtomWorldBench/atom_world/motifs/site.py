from typing import List, Optional

from .base import BaseMotif


class SiteMotif(BaseMotif):
    """A motif that represents a point in space, defined by its coordinates.

    Can be either an atom or an ionic species in a crystal structure.
    """
    allowed_actions = [
        "AddMotifAction",
        "RemoveMotifAction",
        "ReplaceMotifAction",
        "TranslateMotifAction",
        "RotateMotifAction",
        "E3MotifAction",  # E3 operation relative to other motifs or some coordinates.
    ]
    allowed_description_styles = [
        "coord",
        "index",
    ]
    allowed_relative_styles = [
        "centroid_distance",
    ]

    def __init__(
            self,
            *args,
            name: Optional[str] = None,
            indices: Optional[List[int]] = None,
            **kwargs
    ):
        """A Motif is an ASE Atoms comprising a subset of atoms in original ase.Atoms.

        Args:
            *args, **kwargs: See `ase.Atoms.__init__`_.
             .. _ase.Atoms.__init__: https://wiki.fysik.dtu.dk/ase/ase/atoms.html
            name (str, optional): Human-readable motif name. Optional.
             If None, will generate a default name.
            indices (list of int, optional): Original indices from structure.
                Indices should always be provided, if the motif belongs to a specific structure.
        """
        if len(indices) is not None and len(indices) > 1:
            raise ValueError(
                "SiteMotif can only be initialized with a single index."
            )
        super().__init__(*args, name=name, indices=indices, **kwargs)
        if len(self) != 1:
            raise ValueError(
                "SiteMotif must be initialized with exactly one site!"
            )

    def _get_default_name(self) -> str:
        """Generate a default name for the motif based on its species and coordinates."""
        if self.get_initial_charges()[0] == 0:
            return f"an atom {self.species_strings[0]}"
        else:
            return f"a species {self.species_strings[0]}"
