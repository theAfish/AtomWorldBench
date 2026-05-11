"""Active-task action classes.

Active tasks combine *observation* (reading the current structure) with
*action* (modifying it).  Each action class in this package describes:

* The task prompt shown to the model.
* Metadata that the evaluation pipeline needs (e.g. which atom indices were
  added/removed).
* Dataset verifiers to use when scoring model outputs.

Unlike simple/verbose actions these classes do **not** inherit from
``BaseAction`` or ``BaseMotifAction``.  The "action" here is the high-level
task description rather than an atomic structural operation; the actual
structural modifications happen inside the corresponding data generator.
"""

from .remove_molecule import RemoveMoleculeAction

__all__ = ["RemoveMoleculeAction"]
