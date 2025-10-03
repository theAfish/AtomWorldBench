from evaluation.base_evaluator import BaseEvaluator
from models.base_model import BaseModel
from utils.dataloader import load_data
from utils.extract_data import extract_from_string
from prompts.cif_action_prompt import cif_action_prompt
from evaluation.metrics import load_cif_file_from_string, check_atom_counts, match_structures
from typing import Dict, Any
import logging

class AtomWorldEvaluator(BaseEvaluator):
    def __init__(
            self, 
            model: BaseModel,
            data_folder: str, 
            action_name: str = None,
            results_folder: str = "results"
    ):
        """
        Initialize the AtomWorld Evaluator.
        """
        self.data_folder = data_folder
        self.action_name = action_name
        data = load_data(data_folder, action_name)
        super().__init__(model=model, results_folder=results_folder, data=data)

    def _initialize_stats(self) -> Dict:
        """Initialize statistics tracking."""
        return {
            'num_unreadable_out': 0,
            'num_invalid_cif': 0,
            'results': [] 
        }

    def _create_prompt(self, row: Any) -> str:
        """Create a prompt from the data row."""
        if row['input_cif'] is None:
            raise ValueError("input_cif is None")
            
        return cif_action_prompt(
            input_cif=row['input_cif'],
            action_prompt=row['action_prompt'],
            output_format="cif"
        )

    def _process_single_output(
        self,
        row: Any,
        generated_output: str,
        stats: Dict
    ) -> Dict:
        """Process a single generated output."""
        # Extract CIF from output
        generated_cif = extract_from_string(generated_output, format="cif")
        if generated_cif is None:
            logging.info("Invalid generated output")
            stats['num_unreadable_out'] += 1
            return {
                'is_error': True,
                'input_cif': row['input_cif'],
                'action_prompt': row['action_prompt'],
                'generated_output': generated_output,
                'target_cif': row['output_cif'],
                'wrong_type': "OutputFormatError"
            }

        # Parse structures for validation
        output_structure = load_cif_file_from_string(row['output_cif'], primitive=False)
        generated_structure = load_cif_file_from_string(generated_cif, primitive=False)

        if generated_structure is None:
            logging.info("Invalid generated structure")
            stats['num_invalid_cif'] += 1
            return {
                'is_error': True,
                'input_cif': row['input_cif'],
                'action_prompt': row['action_prompt'],
                'generated_output': generated_output,
                'target_cif': row['output_cif'],
                'wrong_type': "CIFParsingError"
            }

        # Check atom counts
        atom_counts_match = check_atom_counts(output_structure, generated_structure)
        if not atom_counts_match:
            logging.info("Atom counts do not match")
            stats['num_invalid_cif'] += 1
            return {
                'is_error': True,
                'input_cif': row['input_cif'],
                'action_prompt': row['action_prompt'],
                'generated_output': generated_cif,
                'target_cif': row['output_cif'],
                'wrong_type': "AtomCountMismatch"
            }

        # Match structures
        rmsd, max_diff = match_structures(output_structure, generated_structure, primitive_cell=False)
        if rmsd == -1:
            logging.info("Structures do not match")
            stats['num_invalid_cif'] += 1
            return {
                'is_error': True,
                'input_cif': row['input_cif'],
                'action_prompt': row['action_prompt'],
                'generated_output': generated_output,
                'target_cif': row['output_cif'],
                'wrong_type': "StructureMismatch"
            }

        # Success case
        return {
            'is_error': False,
            'input_cif': row['input_cif'],
            'action_prompt': row['action_prompt'],
            'generated_cif': generated_cif,
            'target_cif': row['output_cif'],
            'rmsd': rmsd,
            'max_diff': max_diff,
            'generated_output': generated_output
        }

    def _log_success_metrics(self, result: Dict) -> None:
        """Log metrics for successful generations."""
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
        print(f"Evaluation completed. Total inputs: {len(self.data)}")
        print(f"Unreadable outputs: {stats['num_unreadable_out']}, Invalid CIFs: {stats['num_invalid_cif']}")
        
        # Calculate average metrics from successful results
        results = stats.get('results', [])
        if results:
            valid_results = [r for r in results if not r.get('is_error', False)]
            if valid_results:
                avg_rmsd = sum(r['rmsd'] for r in valid_results) / len(valid_results)
                avg_max_diff = sum(r['max_diff'] for r in valid_results) / len(valid_results)
                print(f"Successful results: {len(valid_results)}")
                print(f"Average RMSD: {avg_rmsd}, Average Max Diff: {avg_max_diff}")
            else:
                print("No successful results to compute averages")
        else:
            print("No results found in stats")