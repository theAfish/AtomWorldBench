from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    action_name: Optional[str] = None  # None = all actions in data_folder
    limit: int = -1                    # -1 = all tasks
    repeat: int = 1


class SubmitResultRequest(BaseModel):
    result_cif: str
    elapsed_seconds: Optional[float] = None
    token_usage: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class SessionInfo(BaseModel):
    session_id: str
    status: str            # "pending" | "in_progress" | "evaluated"
    action_name: Optional[str]
    task_count: int
    submitted_count: int
    created_at: str


class TaskListItem(BaseModel):
    task_id: str
    frame_index: int
    repeat_index: int
    action_prompt: str
    action_type: Optional[str]


class TaskDetail(TaskListItem):
    input_cif: str


class SubmitResponse(BaseModel):
    task_id: str
    accepted: bool
    already_submitted: bool


class EvaluationResults(BaseModel):
    session_id: str
    metrics: Dict[str, Any]
    results: List[Dict[str, Any]]
