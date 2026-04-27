
def agent_mode_prompt(
        action_prompt: str,
        workspace_dir: str = "<workspace_dir>",
        output_dir: str = "<output_dir>",
    ) -> str:
    """
    Generate a main prompt for agent-mode evaluation.

    Unlike LLM mode (where the model receives the full CIF content inline and
    outputs a modified CIF as text), agent mode provides the input structure as
    a file on disk and expects the agent to write its result to a file.  The
    agent may use any tools, code execution, or external programs it has access
    to — it is NOT expected to produce a CIF by writing text directly.

    The workspace layout the agent will see:
        {workspace_dir}/
            structure.cif   ← read-only input crystal structure

    The agent must write its result to:
        {output_dir}/
            result.cif      ← modified crystal structure (CIF format)

    Args:
        action_prompt: Natural-language description of the manipulation to
            perform (e.g. "move atom 3 by [1.0, 0.0, 0.0] Angstroms").
        workspace_dir: Path to the workspace directory containing the input CIF.
        output_dir: Path to the output directory where the result CIF should be written.

    Returns:
        A self-contained instruction string suitable for passing to an agent
        as its ``--instruction`` argument.
    """
    lines = [
        "You are an autonomous agent that manipulates crystal structures.",
        "",
        "## Your environment",
        f"- The input crystal structure is stored at: `{workspace_dir}/structure.cif`",
        "  (the exact path is provided to you via the `--workspace_dir` argument).",
        f"- You must write the modified structure to: `{output_dir}/result.cif`",
        "  (the exact path is provided to you via the `--output_dir` argument).",
        "",
        "## How to work",
        "Use code, tools, or any programs available to you to:",
        f"1. Read the input structure from `{workspace_dir}/structure.cif`.",
        "2. Apply the manipulation described below.",
        f"3. Write the modified structure to `{output_dir}/result.cif` in valid CIF format.",
        "",
        "The benchmark only reads `result.cif` from disk.",
        "",
        "## Task",
        action_prompt,
    ]
    return "\n".join(lines)
