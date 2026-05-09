from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


class RegisterUserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    email: Optional[str] = Field(default=None, max_length=256)
    organization: Optional[str] = Field(default=None, max_length=256)


class IssueApiKeyRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    note: Optional[str] = Field(default=None, max_length=256)


class AccessInfoResponse(BaseModel):
    api_name: str
    version: str
    auth: Dict[str, Any]
    endpoints: Dict[str, Any]
    workflow: List[Dict[str, Any]]
    notes: List[str]


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


class RegisteredUserResponse(BaseModel):
    username: str
    email: Optional[str]
    organization: Optional[str]
    created_at: str


class IssuedApiKeyResponse(BaseModel):
    username: str
    email: Optional[str]
    organization: Optional[str]
    api_key: str
    note: Optional[str]
    created_at: str


class EvaluationResults(BaseModel):
    session_id: str
    metrics: Dict[str, Any]
    results: List[Dict[str, Any]]
