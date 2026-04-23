from atomworld.evaluation.base_evaluator import BaseEvaluator
from typing import Dict, Any
import pandas as pd
from utils.extract_data import extract_from_string
from utils.dataloader import load_cif_file_from_string
from prompts.cif_repair_prompt import cif_repair_prompt
from atomworld.evaluation.metrics import check_atom_counts, match_structures
from models.base_model import BaseModel
import logging

class CIFRepairEvaluator(BaseEvaluator):
    def __init__(
            self, 
            model: BaseModel,
            data: pd.DataFrame, 
            results_folder: str = "results"
    ):
        """
        Initialize the CIF Repair Evaluator.
        """
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
        return cif_repair_prompt(modified_cif=row['modified_cif'])

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
                'broken_cif': row['modified_cif'],
                'generated_output': generated_output,
                'fixed_cif': row['original_cif'],
                'wrong_type': "OutputFormatError"
            }

        # Handle value insertions if needed
        if '[VALUE_TO_BE_INSERTED]' in generated_cif and row['removed_value'] != 'None':
            generated_cif = generated_cif.replace('[VALUE_TO_BE_INSERTED]', row['removed_value'])

        # Load and validate structures
        fixed_structure = load_cif_file_from_string(row["original_cif"])
        generated_structure = load_cif_file_from_string(generated_cif)

        if generated_structure is None:
            logging.info("Invalid generated structure")
            stats['num_invalid_cif'] += 1
            return {
                'is_error': True,
                'broken_cif': row['modified_cif'],
                'generated_output': generated_output,
                'fixed_cif': row['original_cif'],
                'wrong_type': "CIFParsingError"
            }

        # Check atom counts
        atom_counts_match = check_atom_counts(fixed_structure, generated_structure)
        if not atom_counts_match:
            logging.info("Atom counts do not match")
            stats['num_invalid_cif'] += 1
            return {
                'is_error': True,
                'broken_cif': row['modified_cif'],
                'generated_output': generated_output,
                'fixed_cif': row['original_cif'],
                'wrong_type': "AtomCountMismatch"
            }

        # Match structures
        rmsd, max_diff = match_structures(fixed_structure, generated_structure)
        if rmsd == -1:
            logging.info("Structures do not match")
            stats['num_invalid_cif'] += 1
            return {
                'is_error': True,
                'broken_cif': row['modified_cif'],
                'generated_output': generated_output,
                'fixed_cif': row['original_cif'],
                'wrong_type': "StructureMismatch"
            }

        # Success case
        return {
            'is_error': False,
            'broken_cif': row['modified_cif'],
            'generated_cif': generated_cif,
            'fixed_cif': row['original_cif'],
            'rmsd': rmsd,
            'max_diff': max_diff,
            'generated_output': generated_output
        }

    def _log_success_metrics(self, result: Dict) -> None:
        """Log metrics for successful generations."""
        print(f"RMSD: {result['rmsd']}, Max Diff: {result['max_diff']}")

    def _print_summary(self, stats: Dict) -> None:
        """Print evaluation summary."""
        print(f"Evaluation completed. Total inputs: {len(self.data)}")
        print(f"Unreadable outputs: {stats['num_unreadable_out']}")
        print(f"Invalid CIFs: {stats['num_invalid_cif']}")
