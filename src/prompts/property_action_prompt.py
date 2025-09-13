
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
        "Propose a specific modification to the structure. This modification must be one or a combination of the following: \n",
        "1. Element Substitution;\n",
        "2. Lattice Parameter Adjustment;\n",
        "3. Atomic Coordinate Adjustment.\n",
        "Please ensure the output is a correct CIF file.",
        "Return the modified CIF content within <cif> and </cif> tags.\n",
        f"Input CIF content:\n{input_cif}\n",
        f"Your goal: modify the CIF file accordingly to {target_trend} the {target_property}."
    ]
    prompt = " ".join(prompt_parts)
    return prompt