import sys
import pytest

# Import the parser class
from benchmark.parser import BenchmarkArgumentParser


def run_with_args(arg_list):
    """Helper to run parser.parse_and_validate with simulated argv"""
    # Backup original argv
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + arg_list
        return BenchmarkArgumentParser.parse_and_validate()
    finally:
        sys.argv = old_argv


def test_atomworld_requires_action(monkeypatch):
    # No action provided for atomworld should raise SystemExit from argparse
    with pytest.raises(SystemExit):
        run_with_args(["-t", "atomworld"])


def test_atomworld_invalid_action(monkeypatch):
    # Invalid action provided should raise SystemExit
    with pytest.raises(SystemExit):
        run_with_args(["-t", "atomworld", "-a", "nonexistent_action"])


def test_atomworld_custom_action_allowed_with_data_path(tmp_path):
    custom_path = str(tmp_path)
    args = run_with_args([
        "-t", "atomworld",
        "-a", "insert_between_atoms_action_natoms",
        "--data_path", custom_path,
    ])
    assert args["action"] == "insert_between_atoms_action_natoms"
    assert args["data_path"] == custom_path


def test_atomworld_valid_action():
    # Valid action should parse and return dict containing provided values
    args = run_with_args(["-t", "atomworld", "-a", "add_atom_action", "-m", "deepseek_chat", "-b", "10"]) 
    assert args["benchmark_type"] == "atomworld"
    assert args["action"] == "add_atom_action"
    assert args["model"] == "deepseek_chat"
    assert args["batch_size"] == 10


def test_pointworld_requires_action():
    with pytest.raises(SystemExit):
        run_with_args(["-t", "pointworld"]) 


def test_cifgen_no_action_allowed():
    # Providing action for cifgen should error
    with pytest.raises(SystemExit):
        run_with_args(["-t", "cifgen", "-a", "some_action"]) 


def test_input_cifs_only_valid_for_atomworld():
    with pytest.raises(SystemExit):
        run_with_args(["-t", "cifgen", "--input_cifs_file", "custom.hdf5"])


def test_default_values_for_missing_optional_args():
    args = run_with_args(["-t", "cifgen"]) 
    # Check some defaults
    assert args["model"] == "deepseek_chat"
    assert args["config"] == "models"
    assert args["batch_size"] == 50
    assert args["num_batch"] == -1
    assert args["restart_from_index"] == 0
    assert args["plot"] is False
    assert args["results_folder"] is None
    assert args["data_path"] is None
    assert args["input_cifs_file"] is None
