import datetime
from pathlib import Path
from typing import Any
from utils.load_model import load_model, load_config
from evaluation.evaluator import AtomWorldEvaluator
from perceptual.evaluator.cif_gen_evaluator import CIFGenEvaluator
from perceptual.evaluator.cif_repair_evaluator import CIFRepairEvaluator
from point_world.evaluator import PointWorldEvaluator
from perceptual.utils.dataloader import load_cif_gen_data, load_data
from .config import BenchmarkConfig

class BenchmarkRunner:
    """Factory class for creating and running different types of benchmarks"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = self._init_model()
        
    def _init_model(self):
        """Initialize the model based on configuration"""
        model_config = load_config(self.config.config_dir / self.config.config_name)[self.config.model_id]
        return load_model(model_config)
    
    def _get_results_folder(self) -> str:
        """Generate results folder path based on benchmark type and configuration"""
        if self.config.results_folder:
            return self.config.results_folder
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        components = [self.config.results_dir, self.config.model_id]
        
        if self.config.action:  # For AtomWorld and PointWorld
            components.append(self.config.action)
            
        components.append(timestamp)
        return str(Path().joinpath(*components))
    
    def _create_atomworld_evaluator(self) -> AtomWorldEvaluator:
        """Create an evaluator for AtomWorld benchmark"""
        data_dir = Path(__file__).parent.parent / "data"
        return AtomWorldEvaluator(
            model=self.model,
            data_folder=data_dir,
            action_name=self.config.action,
            results_folder=self._get_results_folder()
        )
    
    def _create_pointworld_evaluator(self) -> PointWorldEvaluator:
        """Create an evaluator for PointWorld benchmark"""
        data_dir = Path(__file__).parent.parent / "point_world/datasets"
        return PointWorldEvaluator(
            model=self.model,
            data_folder=str(data_dir),
            action_name=self.config.action,
            results_folder=self._get_results_folder()
        )
    
    def _create_cifgen_evaluator(self) -> CIFGenEvaluator:
        """Create an evaluator for CIFGen benchmark"""
        data_dir = Path(__file__).parent.parent / "perceptual/cif_gen/base_cif"
        data = load_cif_gen_data(data_dir)
        return CIFGenEvaluator(
            model=self.model,
            data=data,
            results_folder=self._get_results_folder()
        )
    
    def _create_cifrepair_evaluator(self) -> CIFRepairEvaluator:
        """Create an evaluator for CIFRepair benchmark"""
        data_file = Path(__file__).parent.parent / "perceptual/cif_repair/cif_modifications.csv"
        data = load_data(data_file)
        return CIFRepairEvaluator(
            model=self.model,
            data=data,
            results_folder=self._get_results_folder()
        )
    
    def create_evaluator(self) -> Any:
        """Factory method to create appropriate evaluator based on benchmark type"""
        evaluator_map = {
            'atomworld': self._create_atomworld_evaluator,
            'pointworld': self._create_pointworld_evaluator,
            'cifgen': self._create_cifgen_evaluator,
            'cifrepair': self._create_cifrepair_evaluator
        }
        
        creator = evaluator_map.get(self.config.benchmark_type)
        if not creator:
            raise ValueError(f"Unknown benchmark type: {self.config.benchmark_type}")
            
        return creator()
    
    def run(self):
        """Run the benchmark with current configuration"""
        evaluator = self.create_evaluator()
        evaluator.evaluate(
            batch_size=self.config.batch_size,
            num_batch=self.config.num_batch,
            restart_from_index=self.config.restart_from_index if self.config.restart_from_index else 0
        )