"""Code responsible for operating on atoms."""
from ..common.registry import load_plugins

# Load all atom world plugins (motifs, actions, etc.)
# Silent import.
_ = load_plugins(__package__)
