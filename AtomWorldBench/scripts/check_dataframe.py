import io
import json
from pathlib import Path
from typing import Dict

from ase.io import read, write


def check_one_frame(
    json_data_file: str,
    index: int,
    output_path: str,
    format: str = "cif",
) -> Dict[str, str]:
    """
    Extract one sample from a generator-produced JSON file and write its parts to disk.

    Args:
        json_data_file: Path to the JSON file containing a list of samples.
        index: Zero-based index of the sample to inspect.
        output_path: Directory where the files will be written.
        format: Structure file format to write (e.g., "cif", "xyz").

    Returns:
        Dict with paths to the written files: prompt_path, input_path, output_path.
    """

    data_path = Path(json_data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_data_file}")

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON to contain a list of samples")

    if index < 0 or index >= len(data):
        raise IndexError(f"Index {index} out of range for {len(data)} samples")

    sample = data[index]
    required = ["input", "output", "action_prompt"]
    missing = [k for k in required if k not in sample]
    if missing:
        raise KeyError(f"Sample is missing keys: {', '.join(missing)}")

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{data_path.stem}_{index}"

    prompt_path = out_dir / f"{prefix}_action_prompt.txt"
    prompt_path.write_text(sample["action_prompt"], encoding="utf-8")

    def save_structure(structure_str: str, suffix: str) -> Path:
        # JSON stores CIF strings; we read as CIF then write in the requested format.
        atoms = read(io.StringIO(structure_str), format="cif")
        struct_path = out_dir / f"{prefix}_{suffix}.{format}"
        write(struct_path, atoms, format=format)
        return struct_path

    input_path = save_structure(sample["input"], "input")
    output_path_written = save_structure(sample["output"], "output")

    return {
        "prompt_path": str(prompt_path),
        "input_path": str(input_path),
        "output_path": str(output_path_written),
    }


if __name__ == "__main__":
    data_file = "D:/Codes/AtomWorld/AtomWorldBench/data/RotateStructureAction.json"
    check_one_frame(
        data_file, 1, "D:/Codes/AtomWorld/debug/output_cifs", format='cif'
    )

    from AtomWorldBench.benchmark.evaluation.metrics import compute_exact_match_positional_metrics, match_structures
    from AtomWorldBench.utils.dataloader import load_cif_file_from_string

    with open(data_file, 'r') as f:
        data = json.load(f)[1]  # load the second sample

    input_structure = load_cif_file_from_string(data['input'], primitive=False)
    generated_structure = load_cif_file_from_string(data['output'], primitive=False)
    metrics = match_structures(input_structure, generated_structure)
    print("Is match: ", metrics)

    metrics = compute_exact_match_positional_metrics(input_structure, generated_structure)
    print("Metrics: ", metrics)