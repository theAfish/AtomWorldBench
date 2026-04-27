import os
import shlex
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from benchmark.inference.base_inferencer import BaseInferencer
from benchmark.inference.agent_workspace_materializer import AgentWorkspaceMaterializer
from prompts.agent_mode_prompt import agent_mode_prompt
from utils.dataloader import load_data


class AgentInferencer(BaseInferencer):
    """
    Inferencer that drives an *external* agent CLI instead of a local model.

    For each benchmark task the inferencer:

    1. Materializes a temporary workspace (``structure.cif``).
    2. Spawns the agent subprocess with ``--workspace_dir``, ``--instruction``,
       and ``--output_dir`` arguments.
    3. Reads ``result.cif`` from the agent's output directory.
    4. Wraps the CIF content in ``<cif>…</cif>`` tags so the existing
       :class:`~benchmark.evaluation.atomworld_evaluator.AtomWorldEvaluator`
       can parse it without modification.

    CLI contract expected of the agent::

        <agent_cli> \\
            --workspace_dir <path>   # contains structure.cif
            --instruction   <str>    # natural-language instruction
            --output_dir    <path>   # agent must write result.cif here
    """

    def __init__(
        self,
        agent_cli: str,
        data_folder: str,
        action_name: Optional[str] = None,
        output_folder: str = "inference_outputs",
        timeout: int = 120,
        batch_size: int = 1,
    ):
        """
        Args:
            agent_cli: Shell command (or executable path) for the agent.
                       Supports quoted arguments (parsed with :func:`shlex.split`).
            data_folder: Path to the JSON v2 data folder.
            action_name: If given, only tasks for this action are loaded.
            output_folder: Where inference results and per-task logs are saved.
            timeout: Maximum wall-clock seconds allowed per task.  Tasks that
                     exceed this budget are recorded as failures.
            batch_size: Number of agent subprocesses to run concurrently.
                        ``1`` (default) runs tasks sequentially.  Each worker
                        gets its own isolated temporary workspace and output
                        directory, so parallelism is safe without any locking.
        """
        self.agent_cli = agent_cli
        self.timeout = timeout
        self.workers = max(1, batch_size)
        self._materializer = AgentWorkspaceMaterializer()

        data = load_data(data_folder, action_name)
        if hasattr(data, "to_dict"):
            data = data.to_dict("records")

        # model=None — AgentInferencer never calls model.generate_batch
        super().__init__(model=None, output_folder=output_folder, data=data)

    # ------------------------------------------------------------------
    # BaseInferencer abstract method — required but unused in agent mode
    # ------------------------------------------------------------------

    def _create_prompt(self, row: Any) -> str:
        """Return the instruction string for this task."""
        return agent_mode_prompt(row.get("action_prompt", ""))

    # ------------------------------------------------------------------
    # Core inference loop (fully overrides the LLM-based batch loop)
    # ------------------------------------------------------------------

    def infer(
        self,
        batch_size: int = 1,
        num_batch: int = -1,
        restart_from_index: int = 0,
        repeat: int = 1,
        output_filename: str = "inference_results.json",
    ) -> str:
        """
        Run agent inference over all (or a subset of) tasks.

        ``batch_size`` is accepted for API compatibility with
        :class:`~benchmark.inference.base_inferencer.BaseInferencer` but is
        ignored — agent tasks are always processed one at a time per worker.

        When ``self.workers > 1`` tasks are dispatched concurrently using a
        :class:`~concurrent.futures.ThreadPoolExecutor`.  Each worker drives an
        independent subprocess in its own temporary directory, so parallelism is
        safe without any locking.

        Returns:
            Absolute path to the saved JSON results file.
        """
        if not self.data or len(self.data) <= restart_from_index:
            self.logger.warning(
                "Restart index exceeds data length. No inference performed."
            )
            return None

        if repeat < 1:
            self.logger.warning("Repeat count must be >= 1. Defaulting to 1.")
            repeat = 1

        available_frames = max(len(self.data) - restart_from_index, 0)
        # num_batch acts as a cap on the number of tasks (same semantics as LLM mode)
        target_frames = (
            min(num_batch, available_frames)
            if num_batch > 0
            else available_frames
        )

        self.logger.info(
            f"Starting agent inference: {target_frames} tasks, "
            f"timeout={self.timeout}s per task, repeat={repeat}, "
            f"batch_size (concurrency)={self.workers}."
        )
        self.logger.info(f"Agent CLI: {self.agent_cli}")

        # Build the ordered list of (frame_index, repeat_index, row) to process
        tasks: List[tuple] = []
        for frame_index, row in self._get_data_iterator():
            if frame_index < restart_from_index:
                continue
            for repeat_index in range(repeat):
                tasks.append((frame_index, repeat_index, row))
            if len(tasks) >= target_frames * repeat:
                break

        results = self._run_tasks(tasks)

        output_path = os.path.join(self.output_folder, output_filename)
        self._save_results(results, output_path)
        self.logger.info(f"Agent inference completed. Results saved to {output_path}")
        return output_path

    def _run_tasks(self, tasks: List[tuple]) -> List[Dict]:
        """Execute *tasks* sequentially or in parallel depending on ``self.workers``."""
        if self.workers == 1:
            return self._run_tasks_sequential(tasks)
        return self._run_tasks_parallel(tasks)

    def _run_tasks_sequential(self, tasks: List[tuple]) -> List[Dict]:
        results = []
        for frame_index, repeat_index, row in tqdm(tasks, desc="Agent Inference"):
            generated_output = self._run_agent_task(frame_index, row)
            results.append(
                {
                    "frame_index": frame_index,
                    "repeat_index": repeat_index,
                    "input_data": row,
                    "generated_output": generated_output,
                }
            )
        return results

    def _run_tasks_parallel(self, tasks: List[tuple]) -> List[Dict]:
        """Submit all tasks to a thread pool; preserve original order in output."""
        # Map future → (frame_index, repeat_index, row) so we can reconstruct results
        future_to_meta: Dict = {}
        results: List[Optional[Dict]] = [None] * len(tasks)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            for idx, (frame_index, repeat_index, row) in enumerate(tasks):
                future = executor.submit(self._run_agent_task, frame_index, row)
                future_to_meta[future] = (idx, frame_index, repeat_index, row)

            with tqdm(total=len(tasks), desc=f"Agent Inference (concurrency={self.workers})") as pbar:
                for future in as_completed(future_to_meta):
                    idx, frame_index, repeat_index, row = future_to_meta[future]
                    try:
                        generated_output = future.result()
                    except Exception as exc:
                        self.logger.error(
                            f"Task {frame_index} (repeat {repeat_index}) raised an "
                            f"unexpected exception: {exc}"
                        )
                        generated_output = None
                    results[idx] = {
                        "frame_index": frame_index,
                        "repeat_index": repeat_index,
                        "input_data": row,
                        "generated_output": generated_output,
                    }
                    pbar.update(1)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_agent_task(self, frame_index: int, row: Dict[str, Any]) -> Optional[str]:
        """
        Run the agent for a single task and return its CIF output (or None on
        failure).  The CIF is wrapped in ``<cif>…</cif>`` tags so it is
        compatible with the downstream evaluator.
        """
        log_dir = os.path.join(self.output_folder, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"task_{frame_index}.log")

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_dir = os.path.join(tmp_dir, "workspace")
            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(output_dir)

            # --- materialize workspace ---
            try:
                self._materializer.materialize(row, workspace_dir)
            except Exception as exc:
                self.logger.error(
                    f"Task {frame_index}: workspace materialization failed: {exc}"
                )
                return None

            # --- build command ---
            instruction = self._create_prompt(row)
            cmd = shlex.split(self.agent_cli) + [
                "--workspace_dir", workspace_dir,
                "--instruction", instruction,
                "--output_dir", output_dir,
            ]

            # --- spawn agent ---
            try:
                with open(log_path, "w", encoding="utf-8") as log_file:
                    subprocess.run(
                        cmd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        timeout=self.timeout,
                        check=False,
                    )
            except subprocess.TimeoutExpired:
                self.logger.warning(
                    f"Task {frame_index}: agent timed out after {self.timeout}s."
                )
                return None
            except Exception as exc:
                self.logger.error(f"Task {frame_index}: agent subprocess error: {exc}")
                return None

            # --- read result.cif ---
            result_cif_path = os.path.join(output_dir, "result.cif")
            if not os.path.exists(result_cif_path):
                self.logger.warning(
                    f"Task {frame_index}: result.cif not found in {output_dir}."
                )
                return None

            try:
                with open(result_cif_path, "r", encoding="utf-8") as f:
                    cif_content = f.read().strip()
            except Exception as exc:
                self.logger.error(
                    f"Task {frame_index}: failed to read result.cif: {exc}"
                )
                return None

        # Wrap in tags expected by extract_from_string / AtomWorldEvaluator
        return f"<cif>\n{cif_content}\n</cif>"
