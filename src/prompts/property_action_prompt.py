
# new prompt
def property_action_prompt(
        input_cif: str,
        target_property: str,
        target_trend: str
):
    """
    Generate a prompt for modifing a cif to obtain a desired property change.
    """

    prompt_parts = [
        "You are a material design expert.",
        "Your task is to modify a given CIF file to achieve a desired change in a specific material property.",
        "Please analyze the given CIF file and the target property. "
        "Identify the key structural features and elemental composition that influence the specified property.",
        "Propose a specific modification to the structure. This modification must be one of the following: \n",
        "1. Element substitution is allowed, but no partial occupancy;\n",
        "2. Atomic coordinates may be adjusted flexibly, including long-range collective displacements of atoms, "
        "transformations between different prototype structures, and even changes of the space group symmetry;"
        "For example, a binary compound in the NaCl (FCC) prototype may be reconfigured into the NaCl (simple cubic) prototype.\n"
    ]
    if target_property in ["band gap"]:
        prompt_parts.append("3. Lattice parameter adjustment is allowed.\n")

    prompt_parts += [
        "Please ensure the output is a correct CIF file, and the resulting structure represented by the CIF file should not have any partially occupied site.",
        "Return the modified CIF content within <cif> and </cif> tags.\n",
        f"Input CIF content:\n{input_cif}\n",
        f"Your goal: modify the CIF file accordingly to {target_trend} the {target_property}."
    ]
    prompt = " ".join(prompt_parts)
    return prompt