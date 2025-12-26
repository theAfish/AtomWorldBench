
def cif_action_prompt(
        input_cif: str,
        action_prompt: str,
        output_format: str = "CIF",
):
    """
    Generate a prompt for a CIF action.
    """
    prompt = (
        "You are a crystallography expert.\n"
        "You will receive:\n"
        "1, An input crystal structure in CIF format.\n"
        "2, An action instruction describing one or more operations to perform on this structure.\n"
        "Your task:\n"
        "- Apply the described action(s) to the input structure.\n"
        f"- Return the modified crystal structure in {output_format} format.\n"
        f"- The entire output must be enclosed between the tags"
        f" \"<{output_format}>\" and \"</{output_format}>\", e.g.,"
        f" \"<{output_format}> modified structure content <{output_format}>\".\n"
        "- Do not include any text, comments, or explanations outside these tags.\n"
        "- The output must be a valid crystal structure file with a correct chemical"
        " formula, lattice, and atomic positions.\n"
        "Important notes:\n"
        " 1, Atom indices in the structure start from 0.\n"
        " 2, For multi-step actions:\n"
        "   - After each step, wrap all atomic coordinates back into the unit cell"
        " so that fractional coordinates are within [0, 1].\n"
        "   - Any coordinates mentioned in subsequent steps refer to these wrapped positions.\n"
        f"Input CIF:\n{input_cif}\n"
        f"Action instruction:\n{action_prompt}"
    )
    return prompt