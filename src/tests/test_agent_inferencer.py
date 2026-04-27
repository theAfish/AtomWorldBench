import logging
from pathlib import Path

from benchmark.inference.agent_inferencer import AgentInferencer


def test_run_agent_task_uses_concrete_dirs_in_instruction(tmp_path, monkeypatch):
    inferencer = AgentInferencer.__new__(AgentInferencer)
    inferencer.agent_cli = "fake-agent"
    inferencer.timeout = 5
    inferencer.output_folder = str(tmp_path / "outputs")
    inferencer.logger = logging.getLogger("test-agent-inferencer")

    class FakeMaterializer:
        def materialize(self, row, workspace_dir):
            workspace_path = Path(workspace_dir)
            workspace_path.mkdir(parents=True, exist_ok=True)
            (workspace_path / "structure.cif").write_text("data_test", encoding="utf-8")

    inferencer._materializer = FakeMaterializer()

    captured = {}

    def fake_run(cmd, stdout, stderr, timeout, check):
        instruction = cmd[cmd.index("--instruction") + 1]
        workspace_dir = cmd[cmd.index("--workspace_dir") + 1]
        output_dir = cmd[cmd.index("--output_dir") + 1]

        captured["instruction"] = instruction
        captured["workspace_dir"] = workspace_dir
        captured["output_dir"] = output_dir

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "result.cif").write_text("data_result", encoding="utf-8")

    monkeypatch.setattr("benchmark.inference.agent_inferencer.subprocess.run", fake_run)

    result = inferencer._run_agent_task(
        0,
        {"action_prompt": "Move atom 1 by [0.1, 0.0, 0.0] Angstroms."},
    )

    assert result == "<cif>\ndata_result\n</cif>"
    assert f"{captured['workspace_dir']}/structure.cif" in captured["instruction"]
    assert f"{captured['output_dir']}/result.cif" in captured["instruction"]
    assert "<workspace_dir>" not in captured["instruction"]
    assert "<output_dir>" not in captured["instruction"]