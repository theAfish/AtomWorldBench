import pytest

import os
from pathlib import Path
from ase.io import read

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

atoms_from_cifs = [read(f, index=0) for f in Path(f"{DATA_DIR}/cifs").glob("*.cif")]


@pytest.fixture(params=atoms_from_cifs, scope="package")
def orig_atoms(request):
    """Fixture to provide original atoms from CIF files."""
    return request.param