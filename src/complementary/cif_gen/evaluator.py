from complementary.evaluation.base_evaluator import BaseEvaluator
from typing import Union, List, Dict, Any
import pandas as pd
from utils.extract_data import extract_from_string
from utils.dataloader import load_cif_file_from_string
from complementary.evaluation.metrics import match_structures
from pymatgen.core.structure import Structure
from models.base_model import BaseModel
import logging

class CIFGenEvaluator(BaseEvaluator):
    def __init__(
            self, 
            model: BaseModel,
            data: Union[pd.DataFrame, List[Structure]], 
            results_folder: str = "results"
    ):
        """
        Initialize the CIF Generation Evaluator.
        """
        super().__init__(model=model, results_folder=results_folder, data=data)
        if isinstance(self.data, pd.DataFrame):
            self.data['cif'] = self.data['structure'].apply(lambda s: s.to(fmt="cif"))
            
    def _initialize_stats(self) -> Dict:
        """Initialize statistics tracking."""
        return {
            'num_unreadable_out': 0,
            'num_invalid_cif': 0,
            'results': []  # Initialize results list
        }

    def _create_prompt(self, row: Any) -> str:
        """Create a prompt from the data row."""
        return row['prompt']

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
                'reference_cif': row['cif'],
                'generated_output': generated_output,
                'wrong_type': "OutputFormatError"
            }

        # Parse generated structure
        generated_structure = load_cif_file_from_string(generated_cif)
        if generated_structure is None:
            logging.info("Invalid generated structure")
            stats['num_invalid_cif'] += 1
            return {
                'is_error': True,
                'reference_cif': row['cif'],
                'generated_output': generated_output,
                'wrong_type': "CIFParsingError"
            }

        # Match structures
        rmsd, max_diff = match_structures(row['structure'], generated_structure, primitive_cell=True, attempt_supercell=True)
        if rmsd == -1:
            logging.info("Structures do not match")
            stats['num_invalid_cif'] += 1
            return {
                'is_error': True,
                'reference_cif': row['cif'],
                'generated_output': generated_output,
                'wrong_type': "StructureMismatch"
            }

        # Success case
        return {
            'is_error': False,
            'reference_cif': row['cif'],
            'generated_cif': generated_cif,
            'rmsd': rmsd,
            'max_diff': max_diff,
            'generated_output': generated_output
        }
    
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

    def _log_success_metrics(self, result: Dict) -> None:
        """Log metrics for successful generations."""
        print(f"RMSD: {result['rmsd']}, Max Diff: {result['max_diff']}")

    def _print_summary(self, stats: Dict) -> None:
        """Print evaluation summary."""
        print(f"Evaluation completed. Total inputs: {len(self.data)}")
        print(f"Unreadable outputs: {stats['num_unreadable_out']}")
        print(f"Invalid CIFs: {stats['num_invalid_cif']}")