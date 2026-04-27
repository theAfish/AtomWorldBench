import os
import json
import shlex
import shutil
import subprocess
import tempfile
import time
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

    1. Creates an isolated temporary working directory and materializes
       ``structure.cif`` directly inside it.
    2. Spawns the agent subprocess with ``--instruction``.  The process
       working directory is the task directory, so the agent sees
       ``structure.cif`` directly in its cwd.
    3. After the subprocess exits, searches the task directory recursively
       for the output file (default: ``result.cif``).
    4. Persists the found file to ``<output_folder>/generated_cifs`` so the
       evaluator can load directly from file in agent mode.

    CLI contract expected of the agent::

        <agent_cli> \\
            --instruction   <str>    # natural-language instruction

    The agent's cwd is the task root.  Input is ``structure.cif`` there.
    The agent may write the output file anywhere inside the task root
    (including sub-directories); the benchmark finds it recursively.
    """

    def __init__(
        self,
        agent_cli: str,
        data_folder: str,
        action_name: Optional[str] = None,
        output_folder: str = "inference_outputs",
        timeout: int = 120,
        batch_size: int = 1,
        keep_tmp_workspaces: bool = False,
        output_filename: str = "result.cif",
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
                        gets its own isolated temporary working directory,
                        so parallelism is safe without any locking.
            keep_tmp_workspaces: If True, retain per-task temporary directories
                        under ``<output_folder>/tmp_workspaces`` for debugging.
            output_filename: Filename the agent must produce (default:
                        ``result.cif``).  The benchmark searches for this name
                        recursively inside the task working directory.
        """
        self.agent_cli = agent_cli
        self.inference_mode = "agent"
        self.launch_cwd = os.getcwd()
        self.timeout = timeout
        self.workers = max(1, batch_size)
        self.keep_tmp_workspaces = keep_tmp_workspaces
        self.output_filename = output_filename
        self.tmp_workspace_root = os.path.join(output_folder, "tmp_workspaces")
        self._materializer = AgentWorkspaceMaterializer()

        if self.keep_tmp_workspaces:
            os.makedirs(self.tmp_workspace_root, exist_ok=True)

        data = load_data(data_folder, action_name)
        if hasattr(data, "to_dict"):
            data = data.to_dict("records")

        # model=None — AgentInferencer never calls model.generate_batch
        super().__init__(model=None, output_folder=output_folder, data=data)

    # ------------------------------------------------------------------
    # BaseInferencer abstract method — required but unused in agent mode
    # ------------------------------------------------------------------

    def _create_prompt(
        self,
        row: Any,
        output_filename: str = "result.cif",
    ) -> str:
        """Return the instruction string for this task."""
        return agent_mode_prompt(
            row.get("action_prompt", ""),
            output_filename=output_filename,
        )

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
            task_result = self._run_agent_task_details(frame_index, row, repeat_index)
            results.append(
                {
                    "frame_index": frame_index,
                    "repeat_index": repeat_index,
                    "input_data": row,
                    "inference_mode": self.inference_mode,
                    **task_result,
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
                future = executor.submit(
                    self._run_agent_task_details,
                    frame_index,
                    row,
                    repeat_index,
                )
                future_to_meta[future] = (idx, frame_index, repeat_index, row)

            with tqdm(total=len(tasks), desc=f"Agent Inference (concurrency={self.workers})") as pbar:
                for future in as_completed(future_to_meta):
                    idx, frame_index, repeat_index, row = future_to_meta[future]
                    try:
                        task_result = future.result()
                    except Exception as exc:
                        self.logger.error(
                            f"Task {frame_index} (repeat {repeat_index}) raised an "
                            f"unexpected exception: {exc}"
                        )
                        task_result = {
                            "generated_output": None,
                            "generated_cif_path": None,
                            "agent_status": "internal_error",
                            "agent_elapsed_seconds": None,
                            "agent_return_code": None,
                            "token_usage": None,
                            "agent_usage_source": None,
                        }
                    results[idx] = {
                        "frame_index": frame_index,
                        "repeat_index": repeat_index,
                        "input_data": row,
                        "inference_mode": self.inference_mode,
                        **task_result,
                    }
                    pbar.update(1)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_agent_task(self, frame_index: int, row: Dict[str, Any]) -> Optional[str]:
        """
        Compatibility wrapper that returns only generated_output.

        Tests and legacy callers rely on this return type.
        """
        return self._run_agent_task_details(frame_index, row, 0).get("generated_output")

    def _resolve_agent_cli_tokens(self, cmd_tokens: List[str]) -> List[str]:
        """Resolve relative executable/script tokens against benchmark launch cwd."""
        if not cmd_tokens:
            return cmd_tokens

        resolved = list(cmd_tokens)
        launch_cwd = getattr(self, "launch_cwd", os.getcwd())

        # Resolve relative executable path when possible.
        exe = resolved[0]
        if not os.path.isabs(exe):
            exe_abs = os.path.abspath(os.path.join(launch_cwd, exe))
            if os.path.exists(exe_abs):
                resolved[0] = exe_abs

        # For python-like launchers, resolve the script argument if it is relative.
        if len(resolved) >= 2:
            exe_name = os.path.basename(resolved[0]).lower()
            script = resolved[1]
            if (
                "python" in exe_name
                and script.endswith(".py")
                and not os.path.isabs(script)
            ):
                script_abs = os.path.abspath(os.path.join(launch_cwd, script))
                if os.path.exists(script_abs):
                    resolved[1] = script_abs

        return resolved

    def _run_agent_task_details(
        self,
        frame_index: int,
        row: Dict[str, Any],
        repeat_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Run the agent for a single task and return detailed task metadata.

        The task result includes both a persisted ``generated_cif_path`` (for
        file-based agent evaluation) and ``generated_output`` for compatibility.
        """
        log_dir = os.path.join(self.output_folder, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir,
            f"task_{frame_index}_repeat_{repeat_index}.log",
        )
        instruction_path = os.path.join(
            log_dir,
            f"task_{frame_index}_repeat_{repeat_index}.instruction.txt",
        )

        task_result: Dict[str, Any] = {
            "generated_output": None,
            "generated_cif_path": None,
            "agent_status": "unknown",
            "agent_elapsed_seconds": None,
            "agent_return_code": None,
            "token_usage": None,
            "agent_usage_source": None,
            "agent_log_path": log_path,
            "agent_context_log_path": None,
            "instruction_path": instruction_path,
        }

        keep_tmp_workspaces = getattr(self, "keep_tmp_workspaces", False)
        tmp_workspace_root = getattr(
            self,
            "tmp_workspace_root",
            os.path.join(self.output_folder, "tmp_workspaces"),
        )

        if keep_tmp_workspaces:
            tmp_dir = tempfile.mkdtemp(
                prefix=f"task_{frame_index}_",
                dir=tmp_workspace_root,
            )
            tmp_ctx = None
        else:
            tmp_ctx = tempfile.TemporaryDirectory()
            tmp_dir = tmp_ctx.name

        task_result["tmp_dir"] = tmp_dir if keep_tmp_workspaces else None

        try:
            # --- materialize input CIF directly into workdir root ---
            try:
                self._materializer.materialize(row, tmp_dir)
            except Exception as exc:
                self.logger.error(
                    f"Task {frame_index}: workspace materialization failed: {exc}"
                )
                task_result["agent_status"] = "workspace_materialization_failed"
                return task_result

            # --- build command ---
            output_filename = getattr(self, "output_filename", "result.cif")
            instruction = self._create_prompt(row, output_filename=output_filename)

            with open(instruction_path, "w", encoding="utf-8") as instruction_file:
                instruction_file.write(instruction)

            cmd = self._resolve_agent_cli_tokens(shlex.split(self.agent_cli)) + [
                "--instruction", instruction,
            ]

            # --- spawn agent ---
            try:
                start = time.perf_counter()
                with open(log_path, "w", encoding="utf-8") as log_file:
                    completed = subprocess.run(
                        cmd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        timeout=self.timeout,
                        check=False,
                        cwd=tmp_dir,
                    )
                task_result["agent_elapsed_seconds"] = round(time.perf_counter() - start, 6)
                task_result["agent_return_code"] = getattr(completed, "returncode", 0)
            except subprocess.TimeoutExpired:
                self.logger.warning(
                    f"Task {frame_index}: agent timed out after {self.timeout}s."
                )
                task_result["agent_status"] = "timeout"
                return task_result
            except Exception as exc:
                self.logger.error(f"Task {frame_index}: agent subprocess error: {exc}")
                task_result["agent_status"] = "subprocess_error"
                return task_result

            # Optional usage payload written by external agent for cost tracking.
            usage_path = self._find_file_in_dir(tmp_dir, "usage.json")
            if usage_path is not None:
                try:
                    with open(usage_path, "r", encoding="utf-8") as usage_file:
                        task_result["token_usage"] = json.load(usage_file)
                    task_result["agent_usage_source"] = usage_path
                except Exception as exc:
                    self.logger.warning(
                        f"Task {frame_index}: failed to read usage.json: {exc}"
                    )

            # --- find result.cif (recursive search) ---
            result_cif_path = self._find_file_in_dir(tmp_dir, output_filename)
            if result_cif_path is None:
                self.logger.warning(
                    f"Task {frame_index}: {output_filename!r} not found anywhere in {tmp_dir}."
                )
                task_result["agent_status"] = "missing_result_cif"
                return task_result

            try:
                with open(result_cif_path, "r", encoding="utf-8") as f:
                    cif_content = f.read().strip()
            except Exception as exc:
                self.logger.error(
                    f"Task {frame_index}: failed to read result.cif: {exc}"
                )
                task_result["agent_status"] = "result_read_failed"
                return task_result

            generated_cif_dir = os.path.join(self.output_folder, "generated_cifs")
            os.makedirs(generated_cif_dir, exist_ok=True)
            persisted_result_cif_path = os.path.join(
                generated_cif_dir,
                f"task_{frame_index}_repeat_{repeat_index}.cif",
            )

            try:
                shutil.copyfile(result_cif_path, persisted_result_cif_path)
            except Exception as exc:
                self.logger.error(
                    f"Task {frame_index}: failed to persist result.cif: {exc}"
                )
                task_result["agent_status"] = "result_persist_failed"
                return task_result

            task_result["generated_output"] = f"<cif>\n{cif_content}\n</cif>"
            task_result["generated_cif_path"] = persisted_result_cif_path
            task_result["agent_status"] = "ok"

            if keep_tmp_workspaces:
                self.logger.info(
                    f"Task {frame_index}: preserved temporary workspace at {tmp_dir}"
                )
        finally:
            agent_context_log_path = self._find_file_in_dir(tmp_dir, "agent_context.log")
            if agent_context_log_path is not None:
                persisted_context_log_path = os.path.join(
                    log_dir,
                    f"task_{frame_index}_repeat_{repeat_index}.agent_context.log",
                )
                try:
                    shutil.copyfile(agent_context_log_path, persisted_context_log_path)
                    task_result["agent_context_log_path"] = persisted_context_log_path
                except Exception as exc:
                    self.logger.warning(
                        f"Task {frame_index}: failed to persist agent_context.log: {exc}"
                    )

            if tmp_ctx is not None:
                tmp_ctx.cleanup()

        return task_result

    @staticmethod
    def _find_file_in_dir(root: str, filename: str) -> Optional[str]:
        """
        Walk *root* recursively and return the path of the first file whose
        name matches *filename*, or ``None`` if not found.

        Hidden directories (names starting with ``'.'``) are skipped to avoid
        traversing sandbox-control directories that agents should not see.
        """
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden dirs in-place to avoid traversing sandbox control dirs.
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            if filename in filenames:
                return os.path.join(dirpath, filename)
        return None
