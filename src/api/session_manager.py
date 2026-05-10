"""
Session manager for the AtomWorldBench REST API.

Each benchmark run is a *session*. Sessions are persisted to disk under
``sessions_dir/{session_id}/session.json`` so they survive server restarts.
Submitted result CIF files are stored under
``sessions_dir/{session_id}/results/generated_cifs/``.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional

from api.models import (
    CreateSessionRequest,
    EvaluationResults,
    SessionInfo,
    SubmitResponse,
    SubmitResultRequest,
    TaskDetail,
    TaskListItem,
)


class SessionNotFoundError(Exception):
    pass


class TaskNotFoundError(Exception):
    pass


class SessionManager:
    def __init__(self, data_root: str, sessions_dir: str) -> None:
        self.data_root = data_root
        self.sessions_dir = sessions_dir
        os.makedirs(sessions_dir, exist_ok=True)
        # Per-session locks to prevent concurrent write races
        self._locks: Dict[str, Lock] = {}
        self._global_lock = Lock()

    # ------------------------------------------------------------------
    # Dataset discovery
    # ------------------------------------------------------------------

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Return metadata for every dataset (subdirectory) under data_root."""
        from utils.dataloader import load_data

        datasets = []
        try:
            entries = sorted(os.listdir(self.data_root))
        except OSError:
            return datasets

        for name in entries:
            folder = os.path.join(self.data_root, name)
            if not os.path.isdir(folder):
                continue
            # Count JSON files (= action files)
            json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
            if not json_files:
                continue
            try:
                data = load_data(folder, action_name=None)
                rows = data.to_dict("records") if hasattr(data, "to_dict") else list(data)
                task_count = len(rows)
            except Exception:
                task_count = -1
            datasets.append(
                {
                    "name": name,
                    "action_count": len(json_files),
                    "task_count": task_count,
                }
            )
        return datasets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self, req: CreateSessionRequest) -> SessionInfo:
        from utils.dataloader import load_data

        # Resolve dataset folder
        dataset_name = (req.dataset or "simple").strip()
        data_folder = os.path.join(self.data_root, dataset_name)
        if not os.path.isdir(data_folder):
            raise ValueError(
                f"Dataset {dataset_name!r} not found. "
                f"Available datasets: {[d['name'] for d in self.list_datasets()]}"
            )

        data = load_data(data_folder, req.action_name)
        if hasattr(data, "to_dict"):
            rows = data.to_dict("records")
        else:
            rows = list(data)

        if not rows:
            raise ValueError(
                f"No tasks found for action_name={req.action_name!r} "
                f"in dataset={dataset_name!r}"
            )

        # Apply limit
        if req.limit > 0:
            rows = rows[: req.limit]

        tasks: List[Dict[str, Any]] = []
        for frame_index, row in enumerate(rows):
            for repeat_index in range(max(1, req.repeat)):
                task_id = f"task_{frame_index}_repeat_{repeat_index}"
                tasks.append(
                    {
                        "task_id": task_id,
                        "frame_index": frame_index,
                        "repeat_index": repeat_index,
                        "action_prompt": row.get("action_prompt", ""),
                        "action_type": req.action_name,
                        "input_cif": row.get("input_cif", ""),
                        "output_cif": row.get("output_cif", ""),
                        "submitted": False,
                        "result_cif_path": None,
                        "elapsed_seconds": None,
                        "token_usage": None,
                    }
                )

        session_id = uuid.uuid4().hex
        session_data: Dict[str, Any] = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset_name,
            "action_name": req.action_name,
            "status": "pending",
            "tasks": tasks,
        }

        session_dir = os.path.join(self.sessions_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        self._save_session(session_id, session_data)

        return self._to_session_info(session_data)

    def get_session(self, session_id: str) -> SessionInfo:
        data = self._load_session(session_id)
        return self._to_session_info(data)

    def list_tasks(
        self,
        session_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> List[TaskListItem]:
        data = self._load_session(session_id)
        tasks = data["tasks"][offset : offset + limit]
        return [
            TaskListItem(
                task_id=t["task_id"],
                frame_index=t["frame_index"],
                repeat_index=t["repeat_index"],
                action_prompt=t["action_prompt"],
                action_type=t.get("action_type"),
            )
            for t in tasks
        ]

    def get_task(self, session_id: str, task_id: str) -> TaskDetail:
        task = self._find_task(session_id, task_id)
        return TaskDetail(
            task_id=task["task_id"],
            frame_index=task["frame_index"],
            repeat_index=task["repeat_index"],
            action_prompt=task["action_prompt"],
            action_type=task.get("action_type"),
            input_cif=task["input_cif"],
        )

    def submit_result(
        self,
        session_id: str,
        task_id: str,
        req: SubmitResultRequest,
    ) -> SubmitResponse:
        lock = self._get_lock(session_id)
        with lock:
            data = self._load_session(session_id)
            task = self._find_task_in_data(data, task_id)

            already_submitted = task["submitted"]

            # Persist the result CIF to disk
            cif_dir = os.path.join(
                self.sessions_dir, session_id, "results", "generated_cifs"
            )
            os.makedirs(cif_dir, exist_ok=True)
            cif_path = os.path.join(cif_dir, f"{task_id}.cif")
            with open(cif_path, "w", encoding="utf-8") as f:
                f.write(req.result_cif)

            task["submitted"] = True
            task["result_cif_path"] = cif_path
            task["elapsed_seconds"] = req.elapsed_seconds
            task["token_usage"] = req.token_usage

            # Advance session status
            if data["status"] == "pending":
                data["status"] = "in_progress"

            self._save_session(session_id, data)

        return SubmitResponse(
            task_id=task_id,
            accepted=True,
            already_submitted=already_submitted,
        )

    def run_evaluation(self, session_id: str) -> EvaluationResults:
        from benchmark.evaluation.atomworld_evaluator import AtomWorldEvaluator

        lock = self._get_lock(session_id)
        with lock:
            data = self._load_session(session_id)

        results_dir = os.path.join(self.sessions_dir, session_id, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Build the inference_results.json that AtomWorldEvaluator expects
        inference_results = self._build_inference_results(data)
        inference_path = os.path.join(results_dir, "inference_results.json")
        with open(inference_path, "w", encoding="utf-8") as f:
            json.dump(inference_results, f)

        evaluator = AtomWorldEvaluator(
            action_name=data.get("action_name"),
            results_folder=results_dir,
            inference_mode="agent",
        )
        evaluator.evaluate(inference_path)

        # Mark session as evaluated
        with lock:
            data = self._load_session(session_id)
            data["status"] = "evaluated"
            self._save_session(session_id, data)

        eval_results_path = os.path.join(results_dir, "evaluation_results.json")
        with open(eval_results_path, "r", encoding="utf-8") as f:
            eval_output = json.load(f)

        return EvaluationResults(
            session_id=session_id,
            metrics=eval_output.get("metrics", {}),
            results=eval_output.get("results", []),
        )

    def get_results(self, session_id: str) -> EvaluationResults:
        data = self._load_session(session_id)
        if data["status"] != "evaluated":
            raise ValueError("Session has not been evaluated yet.")

        results_path = os.path.join(
            self.sessions_dir, session_id, "results", "evaluation_results.json"
        )
        if not os.path.exists(results_path):
            raise FileNotFoundError(
                f"Evaluation results file not found: {results_path}"
            )

        with open(results_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        return EvaluationResults(
            session_id=session_id,
            metrics=payload.get("metrics", {}),
            results=payload.get("results", []),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_lock(self, session_id: str) -> Lock:
        with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = Lock()
            return self._locks[session_id]

    def _session_path(self, session_id: str) -> str:
        return os.path.join(self.sessions_dir, session_id, "session.json")

    def _load_session(self, session_id: str) -> Dict[str, Any]:
        path = self._session_path(session_id)
        if not os.path.exists(path):
            raise SessionNotFoundError(session_id)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_session(self, session_id: str, data: Dict[str, Any]) -> None:
        path = self._session_path(session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _find_task(self, session_id: str, task_id: str) -> Dict[str, Any]:
        data = self._load_session(session_id)
        return self._find_task_in_data(data, task_id)

    def _find_task_in_data(
        self, data: Dict[str, Any], task_id: str
    ) -> Dict[str, Any]:
        for t in data["tasks"]:
            if t["task_id"] == task_id:
                return t
        raise TaskNotFoundError(task_id)

    def _to_session_info(self, data: Dict[str, Any]) -> SessionInfo:
        submitted = sum(1 for t in data["tasks"] if t["submitted"])
        return SessionInfo(
            session_id=data["session_id"],
            status=data["status"],
            action_name=data.get("action_name"),
            task_count=len(data["tasks"]),
            submitted_count=submitted,
            created_at=data["created_at"],
        )

    def _build_inference_results(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Convert stored session tasks into the inference_results.json format
        expected by AtomWorldEvaluator.
        """
        records = []
        for task in data["tasks"]:
            records.append(
                {
                    "frame_index": task["frame_index"],
                    "repeat_index": task["repeat_index"],
                    "inference_mode": "agent",
                    "generated_output": None,
                    "generated_cif_path": task.get("result_cif_path"),
                    "agent_status": "ok" if task["submitted"] else "missing_result_cif",
                    "agent_elapsed_seconds": task.get("elapsed_seconds"),
                    "agent_return_code": 0 if task["submitted"] else None,
                    "token_usage": task.get("token_usage"),
                    "agent_usage_source": None,
                    "input_data": {
                        "input_cif": task["input_cif"],
                        "action_prompt": task["action_prompt"],
                        "output_cif": task["output_cif"],
                    },
                }
            )
        return records
