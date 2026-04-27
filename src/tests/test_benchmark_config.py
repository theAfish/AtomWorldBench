from benchmark.config import BenchmarkConfig


def _make_config(benchmark_type: str, action: str = None) -> BenchmarkConfig:
    return BenchmarkConfig(
        benchmark_type=benchmark_type,
        model_id="deepseek_chat",
        batch_size=1,
        num_batch=1,
        config_name="models",
        results_folder=None,
        restart_from_index=0,
        action=action,
    )


def test_atomworld_simple_action_routes_to_simple():
    config = _make_config("atomworld", action="add_atom_action")
    assert config.results_dir == config.base_results_dir / "AtomWorld" / "llm" / "simple"


def test_atomworld_verbose_action_routes_to_verbose():
    config = _make_config("atomworld", action="add_motif_action")
    assert config.results_dir == config.base_results_dir / "AtomWorld" / "llm" / "verbose"


def test_atomworld_unknown_action_defaults_to_simple():
    config = _make_config("atomworld", action="some_future_action")
    assert config.results_dir == config.base_results_dir / "AtomWorld" / "llm" / "simple"


def test_non_atomworld_results_dir_unchanged():
    config = _make_config("cifgen")
    assert config.results_dir == config.base_results_dir / "CifGen"
