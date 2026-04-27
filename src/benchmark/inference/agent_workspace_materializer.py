import os
from typing import Dict, Any


class AgentWorkspaceMaterializer:
    """
    Materializes a single benchmark task into the agent's working directory.

    The input CIF is written directly to the root of *workdir*::

        workdir/
            structure.cif   ← input crystal structure

    The instruction string is passed directly to the agent via ``--instruction``
    and is therefore NOT written to disk here.
    """

    def materialize(self, row: Dict[str, Any], workdir: str) -> str:
        """
        Write the input CIF from *row* into *workdir*.

        Args:
            row: A data record containing at least ``input_cif``.
            workdir: An existing (or newly-created) directory where the
                input file will be written.

        Returns:
            The path to the written ``structure.cif`` file.
        """
        os.makedirs(workdir, exist_ok=True)

        input_cif: str = row.get("input_cif", "")
        if not input_cif:
            raise ValueError("Row is missing a non-empty 'input_cif' field.")

        cif_path = os.path.join(workdir, "structure.cif")
        with open(cif_path, "w", encoding="utf-8") as f:
            f.write(input_cif)

        return cif_path
