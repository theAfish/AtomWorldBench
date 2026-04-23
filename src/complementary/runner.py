"""Runner for complementary benchmark tasks (PointWorld, CIFGen, CIFRepair, StructProp)."""

import datetime
import importlib
import sys
from pathlib import Path
from typing import Any

from complementary.point_world.evaluator import PointWorldEvaluator
from complementary.cif_gen.evaluator import CIFGenEvaluator
from complementary.cif_repair.evaluator import CIFRepairEvaluator
from complementary.struct_prop.inferring import PropertyActionInfer
from complementary.utils.dataloader import load_cif_gen_data, load_data
from models.base_model import BaseModel

# Mapping from benchmark_type to results subfolder name
_RESULT_DIRS = {
    'cifgen': 'CifGen',
    'cifrepair': 'CifRepair',
    'pointworld': 'PointWorld',
    'structprop': 'StructPropBench',
}

# Mapping from structprop action/prop name to data filename
_STRUCTPROP_DATA_FILES = {
    'band_gap': 'bandgap_nonmetal.csv',
    'bulk_modulus': 'bulkmodulus_nonmetal.csv',
}


class ComplementaryRunner:
    """Runner for complementary benchmark tasks.

    Handles PointWorld, CIFGen, CIFRepair, and StructProp.
    """

    def __init__(self, model: BaseModel, config):
        self.model = model
        self.config = config

    def _resolve_data_path(self, default_path: Path) -> Path:
        """Return custom data path if set, otherwise the default."""
        return self.config.custom_data_path or default_path

    def _get_results_folder(self) -> str:
        """Compute the results folder path for the current benchmark type."""
        if self.config.results_folder:
            return self.config.results_folder

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_subdir = _RESULT_DIRS.get(self.config.benchmark_type, self.config.benchmark_type)
        base = (
            self.config.base_results_dir
            / result_subdir
            / self.config.model_id
        )
        if self.config.action:
            base = base / self.config.action
        return str(base / timestamp)

    # ------------------------------------------------------------------
    # Evaluator factories
    # ------------------------------------------------------------------

    def _create_pointworld_evaluator(self) -> PointWorldEvaluator:
        data_dir = self._resolve_data_path(
            Path(__file__).parent.parent / "point_world" / "datasets"
        )
        return PointWorldEvaluator(
            model=self.model,
            data_folder=str(data_dir),
            action_name=self.config.action,
            results_folder=self._get_results_folder(),
        )

    def _create_cifgen_evaluator(self) -> CIFGenEvaluator:
        data_dir = self._resolve_data_path(
            Path(__file__).parent.parent / "perceptual" / "cif_gen" / "base_cif"
        )
        data = load_cif_gen_data(data_dir)
        return CIFGenEvaluator(
            model=self.model,
            data=data,
            results_folder=self._get_results_folder(),
        )

    def _create_cifrepair_evaluator(self) -> CIFRepairEvaluator:
        data_file = self._resolve_data_path(
            Path(__file__).parent.parent / "perceptual" / "cif_repair" / "cif_modifications.csv"
        )
        data = load_data(data_file)
        return CIFRepairEvaluator(
            model=self.model,
            data=data,
            results_folder=self._get_results_folder(),
        )

    def _create_structprop_evaluator(self) -> PropertyActionInfer:
        prop = self.config.action
        if prop not in _STRUCTPROP_DATA_FILES:
            raise ValueError(
                f"Unknown StructProp property '{prop}'. "
                f"Valid options: {list(_STRUCTPROP_DATA_FILES)}"
            )
        import pandas as pd
        data_dir = self._resolve_data_path(Path(__file__).parent / "struct_prop")
        data_file = data_dir / _STRUCTPROP_DATA_FILES[prop]
        data = pd.read_csv(data_file)
        return PropertyActionInfer(
            model=self.model,
            data=data,
            results_folder=self._get_results_folder(),
        )

    def create_evaluator(self) -> Any:
        """Factory: return the appropriate evaluator for the configured benchmark type."""
        factory_map = {
            'pointworld': self._create_pointworld_evaluator,
            'cifgen': self._create_cifgen_evaluator,
            'cifrepair': self._create_cifrepair_evaluator,
            'structprop': self._create_structprop_evaluator,
        }
        creator = factory_map.get(self.config.benchmark_type)
        if not creator:
            raise ValueError(
                f"Unknown complementary benchmark type: '{self.config.benchmark_type}'. "
                f"Valid types: {list(factory_map)}"
            )
        return creator()

    def run(self) -> None:
        """Run the complementary benchmark and optionally generate plots."""
        evaluator = self.create_evaluator()
        evaluator.evaluate(
            batch_size=self.config.batch_size,
            num_batch=self.config.num_batch,
            restart_from_index=self.config.restart_from_index if self.config.restart_from_index else 0,
            repeat=self.config.repeat,
        )

        if getattr(self.config, 'plot', False):
            self._plot_results()

    def _plot_results(self) -> None:
        """Generate the max_dist plot after evaluation (PointWorld / CIFGen only)."""
        if self.config.benchmark_type not in ('pointworld', 'cifgen'):
            print(f"Plotting not supported for benchmark type: {self.config.benchmark_type}")
            return

        model_name = self.config.model_id
        action_name = self.config.action if self.config.action else None
        if self.config.benchmark_type == 'cifgen':
            action_name = None

        results_base = _RESULT_DIRS.get(self.config.benchmark_type)
        if results_base:
            results_base = self.config.base_results_dir / results_base
        else:
            results_base = self.config.base_results_dir

        scripts_dir = Path(__file__).parent.parent / 'scripts'
        analyze_path = scripts_dir / 'analyze_results.py'

        try:
            if str(Path(__file__).parent.parent) not in sys.path:
                sys.path.append(str(Path(__file__).parent.parent))
            from scripts.analyze_results import generate_max_dist_plot

            out_path = generate_max_dist_plot(
                model_name=model_name,
                action_name=action_name,
                results_base=str(results_base),
                out_name=None,
                show=False,
                quiet=True,
            )
            print(f"Saved plot to {out_path}")
        except Exception:
            try:
                spec = importlib.util.spec_from_file_location(
                    'analyze_results', str(analyze_path)
                )
                analyze_mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = analyze_mod
                spec.loader.exec_module(analyze_mod)
                out_path = analyze_mod.generate_max_dist_plot(
                    model_name=model_name,
                    action_name=action_name,
                    results_base=str(results_base),
                    out_name=None,
                    show=False,
                    quiet=True,
                )
                print(f"Saved plot to {out_path}")
            except Exception as e:
                print(f"Failed to generate plot: {e}")
