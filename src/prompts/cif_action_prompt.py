
def cif_action_prompt(
        initial_cif: str,
        action_prompt: str,
        output_format: str = "cif",
):
    """
    Generate a prompt for a CIF action.
    """
    prompt_parts = []
    prompt_parts.append("You are a CIF operation assistant.")
    prompt_parts.append("You will be given an initial CIF content and an action prompt.")
    prompt_parts.append("Your task is to apply the action described in the action prompt to the initial CIF content.")
    prompt_parts.append("The coordinates in the action are in Cartesian format.\n")
    prompt_parts.append(f"Return the modified CIF content in {output_format} format within \"<{output_format}>\" and \"</{output_format}>\" tags.\n")
    prompt_parts.append(f"Initial CIF content:\n{initial_cif}\n")
    prompt_parts.append(f"Action prompt: {action_prompt} ")
    prompt = " ".join(prompt_parts)
    return prompt