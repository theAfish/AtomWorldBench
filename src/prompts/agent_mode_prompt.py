
def agent_mode_prompt(
    action_prompt: str,
    output_filename: str = "result.cif",
) -> str:
    """
    Generate a main prompt for agent-mode evaluation.

    Unlike LLM mode (where the model receives the full CIF content inline and
    outputs a modified CIF as text), agent mode provides the input structure as
    a file on disk and expects the agent to write its result to a file.  The
    agent may use any tools, code execution, or external programs it has access
    to — it is NOT expected to produce a CIF by writing text directly.

    The agent working directory contains the input file and is also where the
    output should be written (though writing to a subdirectory is allowed):

        structure.cif       ← read-only input crystal structure
        {output_filename}   ← the agent must produce this file

    Args:
        action_prompt: Natural-language description of the manipulation to
            perform (e.g. "move atom 3 by [1.0, 0.0, 0.0] Angstroms").
        output_filename: Filename the agent must produce (default: ``result.cif``).

    Returns:
        A self-contained instruction string suitable for passing to an agent
        as its ``--instruction`` argument.
    """
    lines = [
        "You are an autonomous agent that manipulates crystal structures.",
        "",
        "## Your environment",
        "- Your current working directory is an isolated task directory.",
        "- The input crystal structure is at: `structure.cif` (relative to your working directory).",
        f"- You must produce the modified structure as: `{output_filename}`",
        "  (you may write it directly in the working directory or in a subdirectory).",
        "",
        "## How to work",
        "Use code, tools, or any programs available to you to:",
        "1. Read the input structure from `structure.cif`.",
        "2. Apply the manipulation described below.",
        f"3. Write the modified structure to `{output_filename}` in valid CIF format.",
        "4. Do not depend on absolute filesystem paths.",
        "",
        f"The benchmark will search recursively for `{output_filename}` after you finish.",
        "",
        "## Task",
        action_prompt,
    ]
    return "\n".join(lines)
