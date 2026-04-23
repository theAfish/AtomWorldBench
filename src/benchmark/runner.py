import datetime
from pathlib import Path
from typing import Any
from utils.load_model import load_model, load_config
from atomworld.evaluation.evaluator import AtomWorldEvaluator as LegacyAtomWorldEvaluator
from benchmark.inference.inferencer import AtomWorldInferencer
from benchmark.evaluation.atomworld_evaluator import AtomWorldEvaluator
from perceptual.evaluator.cif_gen_evaluator import CIFGenEvaluator
from perceptual.evaluator.cif_repair_evaluator import CIFRepairEvaluator
from point_world.evaluator import PointWorldEvaluator
from perceptual.utils.dataloader import load_cif_gen_data, load_data
from utils.visualization import plot_metrics_distribution
from .config import BenchmarkConfig
import importlib.util
import sys

class BenchmarkRunner:
    """Factory class for creating and running different types of benchmarks"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = self._init_model()
        
    def _init_model(self):
        """Initialize the model based on configuration"""
        model_config = load_config(self.config.config_dir / self.config.config_name)[self.config.model_id]
        return load_model(model_config)

    def _resolve_data_path(self, default_path: Path) -> Path:
        """Return the custom data path if provided, otherwise the default."""
        return self.config.custom_data_path or default_path
    
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
    
    def _create_atomworld_evaluator(self) -> LegacyAtomWorldEvaluator:
        """Create a legacy evaluator for AtomWorld benchmark (combined inference+eval)."""
        data_dir = self._resolve_data_path(Path(__file__).parent.parent / "data")
        input_cifs_file = self.config.atomworld_input_cifs or "input_cifs.hdf5"
        return LegacyAtomWorldEvaluator(
            model=self.model,
            data_folder=str(data_dir),
            action_name=self.config.action,
            results_folder=self._get_results_folder(),
            input_cifs_filename=input_cifs_file,
        )
    
    def _create_pointworld_evaluator(self) -> PointWorldEvaluator:
        """Create an evaluator for PointWorld benchmark"""
        data_dir = self._resolve_data_path(Path(__file__).parent.parent / "point_world/datasets")
        return PointWorldEvaluator(
            model=self.model,
            data_folder=str(data_dir),
            action_name=self.config.action,
            results_folder=self._get_results_folder()
        )
    
    def _create_cifgen_evaluator(self) -> CIFGenEvaluator:
        """Create an evaluator for CIFGen benchmark"""
        data_dir = self._resolve_data_path(Path(__file__).parent.parent / "perceptual/cif_gen/base_cif")
        data = load_cif_gen_data(data_dir)
        return CIFGenEvaluator(
            model=self.model,
            data=data,
            results_folder=self._get_results_folder()
        )
    
    def _create_cifrepair_evaluator(self) -> CIFRepairEvaluator:
        """Create an evaluator for CIFRepair benchmark"""
        data_file = self._resolve_data_path(
            Path(__file__).parent.parent / "perceptual/cif_repair/cif_modifications.csv"
        )
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
        """Run the benchmark with current configuration.
        
        For AtomWorld, uses the separated inference + evaluation pipeline.
        For other benchmarks, uses the legacy combined evaluator.
        """
        if self.config.benchmark_type == 'atomworld':
            self._run_atomworld()
        else:
            self._run_legacy()

    def _run_atomworld(self):
        """Run AtomWorld benchmark with separated inference + evaluation."""
        results_folder = self._get_results_folder()
        inference_folder = str(Path(results_folder) / "inference")
        evaluation_folder = str(Path(results_folder) / "evaluation")

        data_dir = self._resolve_data_path(Path(__file__).parent.parent / "data")

        # Inference
        inferencer = AtomWorldInferencer(
            model=self.model,
            data_folder=str(data_dir),
            action_name=self.config.action,
            output_folder=inference_folder,
        )
        inference_file = inferencer.infer(
            batch_size=self.config.batch_size,
            num_batch=self.config.num_batch,
            restart_from_index=self.config.restart_from_index or 0,
            repeat=self.config.repeat,
        )

        if not inference_file:
            print("No inference results generated.")
            return

        # Evaluation
        evaluator = AtomWorldEvaluator(
            action_name=self.config.action,
            results_folder=evaluation_folder,
        )
        results = evaluator.evaluate(inference_file)

        # Plot
        if getattr(self.config, "plot", False):
            try:
                plot_metrics_distribution(results, evaluation_folder)
            except Exception as e:
                print(f"Error plotting metrics distribution: {e}")

    def _run_legacy(self):
        """Run benchmark using legacy combined evaluator (PointWorld, CIFGen, CIFRepair)."""
        evaluator = self.create_evaluator()
        evaluator.evaluate(
            batch_size=self.config.batch_size,
            num_batch=self.config.num_batch,
            restart_from_index=self.config.restart_from_index if self.config.restart_from_index else 0,
            repeat=self.config.repeat
        )
        # Optionally generate max_dist histogram after evaluation for supported benchmarks
        if getattr(self.config, "plot", False):
            if self.config.benchmark_type not in ("pointworld", "cifgen"):
                print(f"Plotting not supported for benchmark type: {self.config.benchmark_type}")
                return

            model_name = self.config.model_id
            action_name = self.config.action if self.config.action else None
            # For cifgen, reuse model_id as action placeholder
            if self.config.benchmark_type == "cifgen":
                action_name = None

            results_base = self.config.results_dir

            # Prefer direct import of the helper API
            try:
                # ensure src is importable
                if str(Path(__file__).parent.parent) not in sys.path:
                    sys.path.append(str(Path(__file__).parent.parent))
                from scripts.analyze_results import generate_max_dist_plot

                out_path = generate_max_dist_plot(
                    model_name=model_name,
                    action_name=action_name,
                    results_base=results_base,
                    out_name=None,
                    show=False,
                    quiet=True,
                )
                print(f"Saved plot to {out_path}")
            except Exception:
                # Fallback: load module by path and call the programmatic API
                try:
                    scripts_dir = Path(__file__).parent.parent / "scripts"
                    analyze_path = scripts_dir / "analyze_results.py"
                    spec = importlib.util.spec_from_file_location("analyze_results", str(analyze_path))
                    analyze_mod = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = analyze_mod
                    spec.loader.exec_module(analyze_mod)  # type: ignore
                    out_path = analyze_mod.generate_max_dist_plot(
                        model_name=model_name,
                        action_name=action_name,
                        results_base=results_base,
                        out_name=None,
                        show=False,
                        quiet=True,
                    )
                    print(f"Saved plot to {out_path}")
                except Exception as e:
                    print(f"Failed to generate plot: {e}")