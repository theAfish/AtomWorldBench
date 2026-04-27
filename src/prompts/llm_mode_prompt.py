
def llm_mode_prompt(
        input_cif: str,
        action_prompt: str,
        output_format: str = "cif",
):
    """
    Generate a prompt for a CIF action.
    """
    prompt_parts = []
    prompt_parts.append("You are a CIF operation assistant.")
    prompt_parts.append("You will be given an input CIF content and an action prompt.")
    prompt_parts.append("Your task is to apply the action described in the action prompt to the initial CIF content.")
    prompt_parts.append("The coordinates in the action are in Cartesian format, and the indices of atoms are started from 0.\n")
    prompt_parts.append(f"Return the modified CIF content in {output_format} format within \"<{output_format}>\" and \"</{output_format}>\" tags.\n")
    prompt_parts.append("Please ensure the output is a valid CIF file, with correct formula, and atom positions.")
    prompt_parts.append(f"Input CIF content:\n{input_cif}\n")
    prompt_parts.append(f"Action prompt: {action_prompt} ")
    prompt = " ".join(prompt_parts)
    return prompt