import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import plot_metrics_distribution
from utils.args import get_visualization_parser

def main():
    parser = get_visualization_parser()
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return

    # Determine output folder
    if args.output_folder is None:
        args.output_folder = os.path.dirname(args.input_file)
    
    print(f"Loading results from {args.input_file}...")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    # Extract results list
    if 'results' in data:
        results = data['results']
    elif isinstance(data, list):
        results = data
    else:
        print("Error: Could not find 'results' list in the input file.")
        return

    print(f"Found {len(results)} results.")
    print(f"Plotting metrics distribution to {args.output_folder}...")
    
    try:
        plot_metrics_distribution(results, args.output_folder)
        print("Done.")
    except Exception as e:
        print(f"Error plotting metrics: {e}")

if __name__ == "__main__":
    main()
