import numpy as np
import json

def final_prompt(
    input_points: np.ndarray,
    action_prompt: str,
) -> str:
    points_list = input_points.tolist()
    points_str = json.dumps(points_list)

    prompt_parts = [
        "You are a spatial reasoning expert.",
        "You will be given an initial set of points and an action prompt describing an operation on these points.",
        f"The final modified points after applying the action must be returned inside <answer> and </answer> tags.",
        f"The format inside the tags must exactly match the input points format.",
        "All indices are zero-based.",
        "Please ensure the answer inside <answer> and </answer> tags is parseable and strictly formatted.",
        f"Initial points data:",
        points_str,
        "Action prompt:",
        action_prompt,
    ]

    return "\n".join(prompt_parts)

