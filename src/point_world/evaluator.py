from models.base_model import BaseModel
from point_world.data_io import load_dataset_from_h5
from point_world.prompt import final_prompt

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import re
import os
import logging
import json

logging.captureWarnings(True)
logging.basicConfig(
    filename='run.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Evaluator:
    def __init__(
            self,
            model: BaseModel,
            data_folder: str,
            action_name: str = None,
            results_folder: str = "results",
    ):
        """
        Initialize the Evaluator with the data folder and optional action name.
        """
        self.model = model
        self.data_folder = data_folder
        self.action_name = action_name
        self.data = load_dataset_from_h5(data_folder, action_name)
        self.results_folder = results_folder
        os.makedirs(self.results_folder, exist_ok=True)

    def compute_distance_matrix(self, points1, points2):
        diff = points1[:, None, :] - points2[None, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)
        return dist_mat

    # Hungarian algorithm
    def match_points(self, points_true, points_pred):
        dist_mat = self.compute_distance_matrix(points_true, points_pred)
        row_ind, col_ind = linear_sum_assignment(dist_mat)
        matched_distances = dist_mat[row_ind, col_ind]
        return matched_distances

    def evaluate(self, batch_size: int = 8, num_batch: int = -1):
        results = []
        wrongs = []
        num_unreadable_out = 0
        num_invalid_pred = 0

        prompts = []
        rows = []
        batch_count = 0

        if num_batch > 0:
            total = min(len(self.data), batch_size * num_batch)
        else:
            total = len(self.data)

        for i, row in tqdm(enumerate(self.data), total=total, desc="LLM Calling"):
            points_before = row['state_before']
            action_prompt = row['action_prompt']

            prompt = final_prompt(
                input_points=points_before,
                action_prompt=action_prompt
            )
            prompts.append(prompt)
            rows.append(row)

            if len(prompts) == batch_size or i == len(self.data) - 1:
                generated_outputs = self.model.generate_batch(prompts)
                for j, generated_output in enumerate(generated_outputs):
                    row = rows[j]
                    points_pred = extract_points_from_answer(generated_output)
                    points_true = np.array(row['state_after'])

                    if points_pred is None:
                        logging.info(f"Unreadable. Index: {i - len(prompts) + 1 + j}")
                        num_unreadable_out += 1
                        wrongs.append({
                            "state_before": row['state_before'],
                            "action_prompt": row['action_prompt'],
                            "generated_output": generated_output,
                            "target_state_after": row['state_after'],
                        })
                        continue

                    if points_true.shape != points_pred.shape:
                        logging.info(f"Invalid shape. Index: {i - len(prompts) + 1 + j}")
                        num_invalid_pred += 1
                        wrongs.append({
                            "state_before": row['state_before'],
                            "action_prompt": row['action_prompt'],
                            "generated_output": generated_output,
                            "target_state_after": row['state_after'],
                        })
                        continue

                    matched_distances = self.match_points(points_true, points_pred)
                    mean_error = matched_distances.mean()
                    max_error = matched_distances.max()

                    results.append({
                        "state_before": row['state_before'],
                        "action_prompt": row['action_prompt'],
                        "generated_state_after": points_pred.tolist(),
                        "target_state_after": row['state_after'],
                        "mean_error": mean_error,
                        "max_error": max_error,
                        "generated_output": generated_output
                    })
                    # print(f"Mean Error: {mean_error}, Max Error: {max_error}")

                prompts = []
                rows = []
                batch_count += 1

                if num_batch > 0 and batch_count >= num_batch:
                    break

        print(f"Total: {len(self.data)}")
        print(f"Unreadable: {num_unreadable_out}, Invalid: {num_invalid_pred}")
        avg_max_error = np.mean([r['max_error'] for r in results]) if results else float('inf')
        avg_mean_error = np.mean([r['mean_error'] for r in results]) if results else float('inf')
        print(f"Average Max Error: {avg_max_error}, Average Mean Error: {avg_mean_error}")

        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(self.results_folder, f"{self.action_name}_evaluation_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Saved to {results_csv_path}")

        wrongs_df = pd.DataFrame(wrongs)
        wrongs_csv_path = os.path.join(self.results_folder, f"{self.action_name}_evaluation_wrongs.csv")
        wrongs_df.to_csv(wrongs_csv_path, index=False)


def extract_points_from_answer(text: str) -> np.ndarray | None:
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None

    json_str = match.group(1).strip()
    try:
        points_list = json.loads(json_str)
        points = np.array(points_list)
        return points
    except json.JSONDecodeError:
        return None

