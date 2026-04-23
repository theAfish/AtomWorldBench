import json
import argparse
import sys
import os
import logging
from tqdm import tqdm

# Add src to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.extract_data import extract_from_string
from utils.dataloader import load_cif_file_from_string
from complementary.evaluation.metrics import check_atom_counts, match_structures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_entry(entry):
    instruction = entry.get('instruction', '')
    input_cif = entry.get('input', '')
    target_cif_str = entry.get('output', '')
    response = entry.get('response', '')

    # Extract generated CIF
    generated_cif_str = extract_from_string(response, format="cif")
    
    if generated_cif_str is None:
        # Fallback: try to use the whole response if it looks like a CIF
        if "data_" in response:
             generated_cif_str = response

    result = {
        'instruction': instruction,
        'input': input_cif,
        'target': target_cif_str,
        'response': response,
        'generated_cif': generated_cif_str,
        'is_valid': False,
        'atom_counts_match': False,
        'structure_match': False,
        'rmsd': None,
        'max_diff': None,
        'error': None
    }

    if not generated_cif_str:
        result['error'] = "OutputFormatError"
        return result

    # Load structures
    try:
        target_structure = load_cif_file_from_string(target_cif_str, primitive=False)
    except Exception as e:
        result['error'] = f"TargetCIFParsingError: {e}"
        return result

    try:
        generated_structure = load_cif_file_from_string(generated_cif_str, primitive=False)
    except Exception as e:
        result['error'] = f"GeneratedCIFParsingError: {e}"
        return result

    if generated_structure is None:
        result['error'] = "GeneratedCIFParsingError"
        return result
    
    result['is_valid'] = True

    # Check atom counts
    if check_atom_counts(target_structure, generated_structure):
        result['atom_counts_match'] = True
    else:
        result['error'] = "AtomCountMismatch"
        return result

    # Match structures
    rmsd, max_diff = match_structures(target_structure, generated_structure, primitive_cell=False)
    
    if rmsd != -1:
        result['structure_match'] = True
        result['rmsd'] = rmsd
        result['max_diff'] = max_diff
    else:
        result['error'] = "StructureMismatch"

    return result

def main():
    parser = argparse.ArgumentParser(description="Evaluate JSON results for AtomWorld.")
    parser.add_argument("input_file", help="Path to the input JSON file.")
    parser.add_argument("--output_file", help="Path to save the evaluation results (JSON).", default=None)
    parser.add_argument("--summary_file", help="Path to save the summary (JSON).", default=None)
    
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        logging.error(f"Input file not found: {args.input_file}")
        return

    logging.info(f"Loading data from {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    stats = {
        'total': 0,
        'valid_cif': 0,
        'atom_counts_match': 0,
        'structure_match': 0,
        'errors': {}
    }

    logging.info("Evaluating entries...")
    for entry in tqdm(data):
        stats['total'] += 1
        res = evaluate_entry(entry)
        results.append(res)

        if res['is_valid']:
            stats['valid_cif'] += 1
        
        if res['atom_counts_match']:
            stats['atom_counts_match'] += 1
        
        if res['structure_match']:
            stats['structure_match'] += 1
        
        if res['error']:
            err = res['error'].split(':')[0] # Group by error type
            stats['errors'][err] = stats['errors'].get(err, 0) + 1

    # Calculate aggregate metrics for successful matches
    rmsd_values = [r['rmsd'] for r in results if r['rmsd'] is not None]
    max_diff_values = [r['max_diff'] for r in results if r['max_diff'] is not None]

    if rmsd_values:
        stats['rmsd_mean'] = sum(rmsd_values) / len(rmsd_values)
        stats['rmsd_min'] = min(rmsd_values)
        stats['rmsd_max'] = max(rmsd_values)
    
    if max_diff_values:
        stats['max_diff_mean'] = sum(max_diff_values) / len(max_diff_values)
        stats['max_diff_min'] = min(max_diff_values)
        stats['max_diff_max'] = max(max_diff_values)

    logging.info("Evaluation complete.")
    logging.info(f"Summary: {json.dumps(stats, indent=2)}")

    if args.output_file:
        logging.info(f"Saving results to {args.output_file}")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    
    if args.summary_file:
        logging.info(f"Saving summary to {args.summary_file}")
        with open(args.summary_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
    elif args.output_file:
         # If no summary file provided, save summary to a file next to output file
         summary_path = os.path.splitext(args.output_file)[0] + "_summary.json"
         logging.info(f"Saving summary to {summary_path}")
         with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
