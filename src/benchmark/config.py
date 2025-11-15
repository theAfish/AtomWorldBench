from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

@dataclass
class BenchmarkConfig:
    """Common configuration for all benchmarks"""
    # Benchmark type must come first since it's required without a default
    benchmark_type: str  # 'atomworld', 'cifgen', 'cifrepair', 'pointworld'
    
    # Common parameters
    model_id: str
    batch_size: int
    num_batch: int
    config_name: str
    results_folder: Optional[str]
    restart_from_index: Optional[int]
    
    # Base directories with defaults
    config_dir: Path = Path(__file__).parent.parent / "config"
    base_results_dir: Path = Path(__file__).parent.parent.parent / "results"

    # Optional benchmark-specific configurations
    action: Optional[str] = None  # Required for AtomWorld and PointWorld
    plot: bool = False  # Whether to generate the max_dist plot after evaluation
    
    @property
    def results_dir(self) -> Path:
        """Get the appropriate results directory based on benchmark type"""
        type_to_dir = {
            'atomworld': 'AtomWorld',
            'cifgen': 'CifGen',
            'cifrepair': 'CifRepair',
            'pointworld': 'PointWorld'
        }
        return self.base_results_dir / type_to_dir[self.benchmark_type]

    @classmethod
    def from_args(cls, args: Dict[str, Any]) -> 'BenchmarkConfig':
        """Create a BenchmarkConfig from parsed arguments"""
        return cls(
            model_id=args['model'],
            batch_size=args['batch_size'],
            num_batch=args['num_batch'],
            config_name=args.get('config', 'models'),
            results_folder=args.get('results_folder'),
            restart_from_index=args.get('restart_from_index', 0),
            plot=args.get('plot', False),
            benchmark_type=args['benchmark_type'],
            action=args.get('action'),
        )