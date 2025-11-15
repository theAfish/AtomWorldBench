from evaluation.base_evaluator import BaseEvaluator
from models.base_model import BaseModel
from point_world.data_io import load_dataset_from_h5
from point_world.prompt import final_prompt
from typing import Dict, Any, List
import numpy as np
from scipy.optimize import linear_sum_assignment
import re
import json
import logging

class PointWorldEvaluator(BaseEvaluator):
    def __init__(
            self,
            model: BaseModel,
            data_folder: str,
            action_name: str = None,
            results_folder: str = "results",
    ):
        """
        Initialize the PointWorld Evaluator.
        """
        self.data_folder = data_folder
        self.action_name = action_name
        data = load_dataset_from_h5(data_folder, action_name)
        super().__init__(model=model, results_folder=results_folder, data=data)

    def compute_distance_matrix(self, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between two sets of points."""
        diff = points1[:, None, :] - points2[None, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)
        return dist_mat

    def match_points(self, points_true: np.ndarray, points_pred: np.ndarray) -> np.ndarray:
        """Match points using Hungarian algorithm."""
        dist_mat = self.compute_distance_matrix(points_true, points_pred)
        row_ind, col_ind = linear_sum_assignment(dist_mat)
        matched_distances = dist_mat[row_ind, col_ind]
        return matched_distances

    def _initialize_stats(self) -> Dict:
        """Initialize statistics tracking."""
        return {
            'num_unreadable_out': 0,
            'num_invalid_pred': 0,
            'results': []
        }

    def _create_prompt(self, row: Any) -> str:
        """Create a prompt from the data row."""
        return final_prompt(
            input_points=row['state_before'],
            action_prompt=row['action_prompt']
        )

    def _process_single_output(
        self,
        row: Any,
        generated_output: str,
        stats: Dict
    ) -> Dict:
        """Process a single generated output."""
        points_pred = extract_points_from_answer(generated_output)
        points_true = np.array(row['state_after'])

        if points_pred is None:
            logging.info("Unreadable output")
            stats['num_unreadable_out'] += 1
            return {
                'is_error': True,
                'state_before': row['state_before'],
                'action_prompt': row['action_prompt'],
                'generated_output': generated_output,
                'target_state_after': row['state_after'],
                'wrong_type': "UnreadableOutput"
            }

        if points_true.shape != points_pred.shape:
            logging.info("Invalid prediction shape")
            stats['num_invalid_pred'] += 1
            return {
                'is_error': True,
                'state_before': row['state_before'],
                'action_prompt': row['action_prompt'],
                'generated_output': generated_output,
                'target_state_after': row['state_after'],
                'wrong_type': "InvalidShape"
            }

        matched_distances = self.match_points(points_true, points_pred)
        # Use RMSD (root-mean-square deviation) and max difference to match AtomWorldEvaluator
        rmsd = float(np.sqrt(np.mean(np.square(matched_distances))))
        max_diff = float(matched_distances.max())

        return {
            'is_error': False,
            'state_before': row['state_before'],
            'action_prompt': row['action_prompt'],
            'generated_state_after': points_pred.tolist(),
            'target_state_after': row['state_after'],
            'rmsd': rmsd,
            'max_diff': max_diff,
            'generated_output': generated_output
        }

    def _log_success_metrics(self, result: Dict) -> None:
        """Log metrics for successful generations."""
        # Align messaging with AtomWorldEvaluator
        print(f"RMSD: {result['rmsd']}, Max Diff: {result['max_diff']}")


    def _calculate_result_statistics(self, results):
        """
        Calculate statistical metrics from successful results.
        """
        rmsd_values = [res['rmsd'] for res in results if not res['is_error']]
        max_diff_values = [res['max_diff'] for res in results if not res['is_error']]
        
        stats = {
            'rmsd_mean': sum(rmsd_values) / len(rmsd_values) if rmsd_values else None,
            'rmsd_median': sorted(rmsd_values)[len(rmsd_values)//2] if rmsd_values else None,
            'rmsd_max': max(rmsd_values) if rmsd_values else None,
            'rmsd_min': min(rmsd_values) if rmsd_values else None,
            'max_diff_mean': sum(max_diff_values) / len(max_diff_values) if max_diff_values else None,
            'max_diff_median': sorted(max_diff_values)[len(max_diff_values)//2] if max_diff_values else None,
            'max_diff_max': max(max_diff_values) if max_diff_values else None,
            'max_diff_min': min(max_diff_values) if max_diff_values else None,
        }
        return stats

    def _print_summary(self, stats: Dict) -> None:
        """Print evaluation summary."""
        print(f"Total: {len(self.data)}")
        print(f"Unreadable: {stats['num_unreadable_out']}, Invalid: {stats['num_invalid_pred']}")

        # Calculate average errors from successful results
        if 'results' in stats:
            results = [r for r in stats['results'] if not r.get('is_error', False)]
            if results:
                avg_max_diff = np.mean([r['max_diff'] for r in results])
                avg_rmsd = np.mean([r['rmsd'] for r in results])
                print(f"Average Max Diff: {avg_max_diff}, Average RMSD: {avg_rmsd}")
            else:
                print("No successful results to compute average errors")



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

