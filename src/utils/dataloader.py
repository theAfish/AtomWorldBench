import os
import re
import json
import pandas as pd
from pymatgen.io.cif import CifParser
import logging


def load_cif_file(cif_file, primitive=True):
    """
    Load a CIF file and return the first structure.
    If the file is not valid, return None.
    """
    try:
        parser = CifParser(cif_file)
        structures = parser.parse_structures(primitive=primitive, check_occu=False)
        if structures:
            return structures[0]
        else:
            logging.info(f"No structures found in {cif_file}.")
            return None
    except Exception as e:
        logging.info(f"Error loading CIF file {cif_file}: {e}")
        return None

def load_cif_file_from_string(cif_string, primitive=True):
    """
    Load a CIF file from a string and return the first structure.
    If the string is not valid, return None.
    """
    try:
        parser = CifParser.from_str(cif_string)
        structures = parser.parse_structures(primitive=primitive, check_occu=False)
        if structures:
            return structures[0]
        else:
            logging.info("No structures found in the CIF string.")
            return None
    except Exception as e:
        logging.info(f"Error loading CIF from string: {e}")
        return None
    


def format_floats_in_string(s, precision):
    """
    Format floating-point numbers found in string `s` to given `precision` decimal places.
    Only matches floats (numbers with a decimal point or exponent) and leaves integers alone.
    """
    if s is None or precision is None:
        return s
    # match floats with optional exponent, avoid matching plain integers or
    # numbers that are just an integer followed by a punctuation dot (e.g. "0.")
    # Require at least one digit after the decimal point.
    pattern = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?')

    def repl(m):
        txt = m.group(0)
        try:
            val = float(txt)
        except Exception:
            return txt
        fmt = f"{{:.{precision}f}}"
        return fmt.format(val)

    return pattern.sub(repl, s)


def _normalize_action_name(name):
    """Normalize action-like names into a comparable snake_case form."""
    if not name:
        return ""

    # Keep only the filename stem for comparisons.
    base = os.path.splitext(str(name))[0]
    # Split CamelCase and normalize separators.
    base = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", base)
    base = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", base)
    base = re.sub(r"[^A-Za-z0-9]+", "_", base)
    base = re.sub(r"_+", "_", base)
    return base.strip("_").lower()


def _action_aliases(name):
    """Return equivalent aliases for matching action filenames."""
    normalized = _normalize_action_name(name)
    if not normalized:
        return set()

    aliases = {normalized}
    if normalized.endswith("_action"):
        aliases.add(normalized[:-7])
    else:
        aliases.add(f"{normalized}_action")
    return aliases


def _resolve_action_file(action_name, candidate_files):
    """Resolve an action name to one candidate filename with robust matching."""
    if not action_name:
        return None

    # 1) Direct full filename match (case-sensitive then case-insensitive).
    if action_name in candidate_files:
        return action_name
    lower_to_original = {f.lower(): f for f in candidate_files}
    if action_name.lower() in lower_to_original:
        return lower_to_original[action_name.lower()]

    target_stem = os.path.splitext(action_name)[0]

    # 2) Stem match (case-sensitive then case-insensitive).
    for filename in candidate_files:
        if os.path.splitext(filename)[0] == target_stem:
            return filename
    for filename in candidate_files:
        if os.path.splitext(filename)[0].lower() == target_stem.lower():
            return filename

    # 3) Normalized alias match (snake/camel and optional `_action` suffix).
    target_aliases = _action_aliases(target_stem)
    matched = []
    for filename in candidate_files:
        stem = os.path.splitext(filename)[0]
        if target_aliases & _action_aliases(stem):
            matched.append(filename)

    if not matched:
        return None

    # Prefer shortest exact-ish stem to keep selection deterministic.
    matched.sort(key=lambda f: (len(os.path.splitext(f)[0]), os.path.splitext(f)[0].lower()))
    return matched[0]


def load_data(data_folder, action_name=None, input_cifs="input_cifs.hdf5", precision=None):
    """
    Loads data for training/analysis.
    Automatically detects CSV+HDF5 (v1) or JSON (v2) format.

    Detection logic:
    - If CSV files exist in the folder → use CSV+HDF5 (v1 format)
    - Otherwise, if JSON files exist → use JSON (v2 format)

    Args:
        data_folder: Path to the folder containing data files.
        action_name: If specified, only load data for this action.
        input_cifs: HDF5 filename for input CIFs (CSV+HDF5 mode only).
        precision: Optional int for float formatting.
    Returns:
        pd.DataFrame with columns: input_cif, action_prompt, output_cif
    """
    if not os.path.exists(data_folder):
        logging.error(f"Data folder {data_folder} does not exist.")
        return pd.DataFrame()

    files = os.listdir(data_folder)
    csv_files = [f for f in files if f.endswith(".csv")]
    json_files = [f for f in files if f.endswith(".json")]

    # CSV+HDF5 (v1) takes priority when CSV files exist
    if csv_files:
        return _load_data_from_csv_hdf5(data_folder, action_name, input_cifs, precision)

    # Fall back to JSON (v2 format)
    if json_files:
        return _load_data_from_json(data_folder, action_name, json_files)

    logging.warning(f"No CSV or JSON data files found in {data_folder}")
    return pd.DataFrame()


def _load_data_from_json(data_folder, action_name=None, json_files=None):
    """Load data from per-action JSON files (v2 format)."""
    if json_files is None:
        json_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]

    if action_name:
        target_file = _resolve_action_file(action_name, json_files)
        if target_file is None:
            logging.warning(f"Action {action_name} not found in {data_folder}.")
            return pd.DataFrame()
        if os.path.splitext(target_file)[0] != os.path.splitext(action_name)[0]:
            logging.info(f"Resolved action {action_name} -> {target_file}")
        json_files = [target_file]

    rows = []
    for json_file in json_files:
        json_path = os.path.join(data_folder, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                rows.append({
                    "input_cif": item.get("input", ""),
                    "action_prompt": item.get("action_prompt", ""),
                    "output_cif": item.get("output", ""),
                    # Active-task fields — None for simple/verbose items
                    "task_category": item.get("task_category", None),
                    "verifiers": item.get("verifiers", None),
                    "metadata": item.get("metadata", None),
                })
        except Exception as e:
            logging.error(f"Error loading {json_file}: {e}")

    return pd.DataFrame(rows)


def _load_data_from_csv_hdf5(data_folder, action_name=None, input_cifs="input_cifs.hdf5", precision=None):
    """Load data from CSV+HDF5 files (v1 format)."""
    from scripts.load_data_from_h5 import load_cifs_from_hdf5

    # Load all input CIFs
    input_cifs_path = os.path.join(data_folder, input_cifs)
    input_cifs_dict = load_cifs_from_hdf5(input_cifs_path)

    # Find all action CSVs
    files = os.listdir(data_folder)
    action_csvs = [f for f in files if f.endswith(".csv")]
    if action_name:
        resolved_csv = _resolve_action_file(action_name, action_csvs)
        action_csvs = [resolved_csv] if resolved_csv else []

    rows = []
    for csv_file in action_csvs:
        action_base = csv_file[:-4]
        h5_file = f"{action_base}.hdf5"
        h5_path = os.path.join(data_folder, h5_file)
        # Load output CIFs for this action
        output_cifs = load_cifs_from_hdf5(h5_path) if os.path.exists(h5_path) else {}
        # Read CSV
        csv_path = os.path.join(data_folder, csv_file)
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # input_cif: get from input_cifs dict
            input_cif_path = os.path.basename(row["input_cif"])
            input_cif_str = input_cifs_dict.get(input_cif_path, "")
            # action_prompt: from CSV
            action_prompt = row["action_prompt"]
            # output_cif: get from output_cifs dict
            output_cif_path = os.path.basename(row["output_cif"])
            output_cif_str = output_cifs.get(output_cif_path, "")
            # If precision is provided, format floating point numbers consistently
            if precision is not None:
                try:
                    input_cif_str = format_floats_in_string(input_cif_str, precision)
                except Exception:
                    pass
                try:
                    action_prompt = format_floats_in_string(action_prompt, precision)
                except Exception:
                    pass
                try:
                    output_cif_str = format_floats_in_string(output_cif_str, precision)
                except Exception:
                    pass

            rows.append({
                "input_cif": input_cif_str,
                "action_prompt": action_prompt,
                "output_cif": output_cif_str
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data_folder = "." 
    action_name = 'add_atom_action' 
    # Example: set precision=6 to format floats to 6 decimal places
    df = load_data(data_folder, action_name, precision=6)
    print(df.head()['action_prompt'])