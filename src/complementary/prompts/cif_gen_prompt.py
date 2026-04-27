from pymatgen.io.cif import Structure


def cif_gen_prompt(
        info: dict = None
):
    """
    Generate a prompt for generating a cif.
    """
    formula = info["formula"]
    # try to get the full_name, if no, raise error
    if "structure_type" in info:
        structure_type = info["structure_type"]
    else:
        raise ValueError("info must contain structure_type")
    

    # get info according to must_fields from reference_cif
    info_lines = []
    if "a" in info:
        info_lines.append(f"- Lattice constant a: {info['a']} Angstrom")
    if "b" in info:
        info_lines.append(f"- Lattice constant b: {info['b']} Angstrom")
    if "c" in info:
        info_lines.append(f"- Lattice constant c: {info['c']} Angstrom")
    if "c-aligned atom" in info:
        the_atom = info.get("c-aligned atom", None)
        if the_atom:
            info_lines.append(f"- The inter-layer {the_atom} atoms are aligned along the c axis.")
        else:
            raise ValueError("Must provide c-aligned atom.")
    if "octahedron center" in info:
        center_atom = info.get("octahedron center", None)
        if center_atom:
            info_lines.append(f"- The {center_atom} atom is at the center of the octahedron formed by surrounding atoms.")
        else:
            raise ValueError("Must provide octahedron center.")

    prompt_parts = [
        "You are a materials science expert.",
        "Please generate some simple and standard structures in the CIF format according to the requirements.",
        "You must strictly follow the CIF format specifications.",
        "Since the symmetry-related information can be complex, please write the CIF file with P1 symmetry.",
        "Please ensure the output is a correct CIF file.",
        "Return the fixed CIF content within <cif> and </cif> tags.\n",
        "Requirements:\n",
        f"Please generate a CIF file for {formula} with a {structure_type} structure, according to the following information about the convensional cell:\n",
        "\n ".join(info_lines),
    ]
    prompt = " ".join(prompt_parts)
    return prompt

