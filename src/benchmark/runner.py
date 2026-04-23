import datetime
from pathlib import Path
from utils.load_model import load_model, load_config
from benchmark.inference.inferencer import AtomWorldInferencer
from benchmark.evaluation.atomworld_evaluator import AtomWorldEvaluator
from utils.visualization import plot_metrics_distribution
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
    
    def run(self):
        """Run the benchmark with current configuration.
        
        For AtomWorld, uses the separated inference + evaluation pipeline.
        For other benchmarks, delegates to the complementary runner.
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
        """Run benchmark using complementary runner (PointWorld, CIFGen, CIFRepair, StructProp)."""
        from complementary.runner import ComplementaryRunner
        ComplementaryRunner(self.model, self.config).run()