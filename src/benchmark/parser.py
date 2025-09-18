import argparse
from typing import Dict, Any

class BenchmarkArgumentParser:
    """Unified argument parser for all benchmark types"""
    
    # Available benchmark types and their actions
    BENCHMARK_TYPES = {
        'atomworld': [
            'add_atom_action', 'change_atom_action', 'delete_around_atom_action',
            'delete_below_atom_action', 'insert_between_atoms_action', 'move_around_atom_action',
            'move_atom_action', 'move_selected_atoms_action', 'move_towards_atom_action',
            'remove_atom_action', 'rotate_around_atom_action', 'swap_atoms_action'
        ],
        'pointworld': ['move', 'move_towards', 'insert_between', 'rotate_around'],
        'cifgen': [],  # No specific actions
        'cifrepair': []  # No specific actions
    }

    @classmethod
    def create_parser(cls) -> argparse.ArgumentParser:
        """Create the argument parser with all possible arguments"""
        parser = argparse.ArgumentParser(description="Run benchmark with specified configuration.")
        
        # Common arguments for all benchmark types
        parser.add_argument(
            "-t",
            "--benchmark_type",
            type=str,
            choices=cls.BENCHMARK_TYPES.keys(),
            required=True,
            help="Type of benchmark to run"
        )
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            default="deepseek_chat",
            help="ID of the model to use (e.g., 'deepseek_chat', 'openai_gpt4')"
        )
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            default="models",
            help="Name of config file (located under src/config)"
        )
        parser.add_argument(
            "-b",
            "--batch_size",
            type=int,
            default=50,
            help="Batch size for the run. Default: 50"
        )
        parser.add_argument(
            "-n",
            "--num_batch",
            type=int,
            default=-1,
            help="Number of batches to use. Default: -1 for all data"
        )
        parser.add_argument(
            "-f",
            "--results_folder",
            type=str,
            default=None,
            help="Folder to save results. Will be automatically generated if not provided."
        )
        parser.add_argument(
            "-r",
            "--restart_from_index",
            type=int,
            default=0,
            help="Index to restart from (for long runs that were interrupted)"
        )
        
        # Action argument (only used by some benchmark types)
        parser.add_argument(
            "-a",
            "--action",
            type=str,
            help="Name of the action for test (required for AtomWorld and PointWorld benchmarks)"
        )
        
        return parser
    
    @classmethod
    def parse_and_validate(cls) -> Dict[str, Any]:
        """Parse arguments and validate benchmark-specific requirements"""
        parser = cls.create_parser()
        args = parser.parse_args()
        
        # Convert to dictionary for easier manipulation
        args_dict = vars(args)
        
        # Validate action argument usage
        if args.benchmark_type in ['atomworld', 'pointworld']:
            # Action is required for these benchmark types
            if not args.action:
                parser.error(f"--action is required for {args.benchmark_type} benchmark")
            
            valid_actions = cls.BENCHMARK_TYPES[args.benchmark_type]
            if args.action not in valid_actions:
                parser.error(f"Invalid action '{args.action}' for {args.benchmark_type}. "
                           f"Must be one of: {valid_actions}")
        elif args.action is not None:
            # Action should not be provided for other benchmark types
            parser.error(f"--action argument is not used for {args.benchmark_type} benchmark. "
                        f"This argument is only valid for 'atomworld' and 'pointworld' benchmarks.")
        
        return args_dict