
def cif_repair_prompt(
        modified_cif: str,
):
    """
    Generate a prompt for fixing a cif.
    """
    prompt_parts = [
        "You are a CIF operation assistant.",
        "You will be given a CIF content that may be corrupted or incomplete.",
        "Your task is to examine the CIF content and fix any issues to ensure it is a valid CIF file.",
        "If there are missing values that cannot be repaired directly, you can use the [VALUE_TO_BE_INSERTED] as hints to fill in the missing values.",
        "Please ensure the output is a correct CIF file.",
        "Return the fixed CIF content within <cif> and </cif> tags.",
        f"Input CIF content:\n{modified_cif}\n"
    ]
    prompt = " ".join(prompt_parts)
    return prompt
