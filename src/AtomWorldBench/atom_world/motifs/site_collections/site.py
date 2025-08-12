from .base import BaseSiteCollectionMotif


class SiteMotif(BaseSiteCollectionMotif):
    """A motif that represents a point in space, defined by its coordinates.

    Can be either an atom or an ionic species in a crystal structure.
    """
    allowed_actions = BaseSiteCollectionMotif.allowed_actions
    allowed_actions.pop("rotate") # SiteMotif cannot be rotated.
    allowed_actions.pop("resize")  # SiteMotif cannot be resized.

    allowed_relative_styles = {
        "centroid_distance": None,
    }

    # To supress mypy, we need to define __len__ explicitly.
    def __len__(self) -> int: ...

    def __post_init__(self):
        """Post-initialization to check whether motif size is 1."""
        if len(self) != 1:
            raise ValueError(f"SiteMotif must contain exactly one site, but got {len(self)} sites.")

    def _get_default_name(self) -> str:
        """Generate a default name for the motif based on its species and coordinates."""
        if self.get_initial_charges()[0] == 0:
            return f"an atom {self.species_strings[0]}"
        else:
            return f"a species {self.species_strings[0]}"
