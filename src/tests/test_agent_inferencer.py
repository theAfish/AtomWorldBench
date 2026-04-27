import logging
from pathlib import Path

from benchmark.inference.agent_inferencer import AgentInferencer


def test_run_agent_task_uses_flat_workdir(tmp_path, monkeypatch):
    inferencer = AgentInferencer.__new__(AgentInferencer)
    inferencer.agent_cli = "fake-agent"
    inferencer.timeout = 5
    inferencer.output_folder = str(tmp_path / "outputs")
    inferencer.output_filename = "result.cif"
    inferencer.logger = logging.getLogger("test-agent-inferencer")

    class FakeMaterializer:
        def materialize(self, row, workdir):
            workdir_path = Path(workdir)
            workdir_path.mkdir(parents=True, exist_ok=True)
            (workdir_path / "structure.cif").write_text("data_test", encoding="utf-8")

    inferencer._materializer = FakeMaterializer()

    captured = {}

    def fake_run(cmd, stdout, stderr, timeout, check, cwd):
        instruction = cmd[cmd.index("--instruction") + 1]

        captured["instruction"] = instruction
        captured["cwd"] = cwd
        # Confirm structure.cif is directly in the workdir (flat layout)
        captured["structure_cif_exists"] = (Path(cwd) / "structure.cif").is_file()
        # Confirm no workspace/ or output/ subdirs were created by inferencer
        captured["workspace_subdir_exists"] = (Path(cwd) / "workspace").is_dir()
        captured["output_subdir_exists"] = (Path(cwd) / "output").is_dir()

        # Write result.cif in a subdirectory to test recursive discovery
        subdir = Path(cwd) / "subdir"
        subdir.mkdir()
        (subdir / "result.cif").write_text("data_result", encoding="utf-8")

        class FakeCompleted:
            returncode = 0

        return FakeCompleted()

    monkeypatch.setattr("benchmark.inference.agent_inferencer.subprocess.run", fake_run)

    result = inferencer._run_agent_task(
        0,
        {"action_prompt": "Move atom 1 by [0.1, 0.0, 0.0] Angstroms."},
    )

    assert result == "<cif>\ndata_result\n</cif>"
    assert captured["structure_cif_exists"] is True
    assert captured["workspace_subdir_exists"] is False
    assert captured["output_subdir_exists"] is False
    assert "structure.cif" in captured["instruction"]
    assert "result.cif" in captured["instruction"]
    assert "--workspace_dir" not in captured["instruction"]
    assert "--output_dir" not in captured["instruction"]


def test_find_file_in_dir_skips_hidden_dirs(tmp_path):
    # Create a hidden dir with the file — should be skipped
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "result.cif").write_text("hidden", encoding="utf-8")

    # Create a visible subdir with the file — should be found
    visible = tmp_path / "subdir"
    visible.mkdir()
    (visible / "result.cif").write_text("found", encoding="utf-8")

    found = AgentInferencer._find_file_in_dir(str(tmp_path), "result.cif")
    assert found is not None
    assert Path(found).read_text() == "found"