"""
AtomWorldBench REST API server.

Start with:
    atomworld serve --api-key <key> --data-folder data/simple --sessions-dir sessions

All endpoints require the ``X-API-Key`` header matching the key supplied at startup.
"""

import os
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import JSONResponse

from api.models import (
    CreateSessionRequest,
    EvaluationResults,
    SessionInfo,
    SubmitResponse,
    SubmitResultRequest,
    TaskDetail,
    TaskListItem,
)
from api.session_manager import SessionManager, SessionNotFoundError, TaskNotFoundError


def create_app(data_folder: str, sessions_dir: str) -> FastAPI:
    app = FastAPI(
        title="AtomWorldBench API",
        description=(
            "REST API for running AtomWorld agent benchmarks. "
            "Create a session, fetch tasks, submit results, then evaluate."
        ),
        version="0.1.0",
    )

    manager = SessionManager(data_folder=data_folder, sessions_dir=sessions_dir)

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def require_api_key(x_api_key: Annotated[Optional[str], Header()] = None):
        expected = os.environ.get("ATOMWORLD_API_KEY", "")
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server is not configured with an API key.",
            )
        if x_api_key != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header.",
            )

    Auth = Annotated[None, Depends(require_api_key)]

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------

    def _not_found(detail: str):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

    def _bad_request(detail: str):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    # ------------------------------------------------------------------
    # Session endpoints
    # ------------------------------------------------------------------

    @app.post(
        "/sessions",
        response_model=SessionInfo,
        status_code=status.HTTP_201_CREATED,
        summary="Create a new benchmark session",
    )
    def create_session(_: Auth, req: CreateSessionRequest):
        try:
            return manager.create_session(req)
        except ValueError as exc:
            raise _bad_request(str(exc))

    @app.get(
        "/sessions/{session_id}",
        response_model=SessionInfo,
        summary="Get session status",
    )
    def get_session(_: Auth, session_id: str):
        try:
            return manager.get_session(session_id)
        except SessionNotFoundError:
            raise _not_found(f"Session {session_id!r} not found.")

    # ------------------------------------------------------------------
    # Task endpoints
    # ------------------------------------------------------------------

    @app.get(
        "/sessions/{session_id}/tasks",
        response_model=list[TaskListItem],
        summary="List tasks for a session (no CIF content)",
    )
    def list_tasks(
        _: Auth,
        session_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=500),
    ):
        try:
            return manager.list_tasks(session_id, offset=offset, limit=limit)
        except SessionNotFoundError:
            raise _not_found(f"Session {session_id!r} not found.")

    @app.get(
        "/sessions/{session_id}/tasks/{task_id}",
        response_model=TaskDetail,
        summary="Get task details including input CIF",
    )
    def get_task(_: Auth, session_id: str, task_id: str):
        try:
            return manager.get_task(session_id, task_id)
        except SessionNotFoundError:
            raise _not_found(f"Session {session_id!r} not found.")
        except TaskNotFoundError:
            raise _not_found(f"Task {task_id!r} not found in session {session_id!r}.")

    # ------------------------------------------------------------------
    # Submission endpoint
    # ------------------------------------------------------------------

    @app.post(
        "/sessions/{session_id}/tasks/{task_id}/submit",
        response_model=SubmitResponse,
        summary="Submit the result CIF for a task",
    )
    def submit_result(
        _: Auth,
        session_id: str,
        task_id: str,
        req: SubmitResultRequest,
    ):
        try:
            return manager.submit_result(session_id, task_id, req)
        except SessionNotFoundError:
            raise _not_found(f"Session {session_id!r} not found.")
        except TaskNotFoundError:
            raise _not_found(f"Task {task_id!r} not found in session {session_id!r}.")

    # ------------------------------------------------------------------
    # Evaluation endpoints
    # ------------------------------------------------------------------

    @app.post(
        "/sessions/{session_id}/evaluate",
        response_model=EvaluationResults,
        summary="Trigger evaluation for all submitted tasks",
    )
    def evaluate_session(_: Auth, session_id: str):
        try:
            return manager.run_evaluation(session_id)
        except SessionNotFoundError:
            raise _not_found(f"Session {session_id!r} not found.")
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Evaluation failed: {exc}",
            )

    @app.get(
        "/sessions/{session_id}/results",
        response_model=EvaluationResults,
        summary="Get evaluation results (must call /evaluate first)",
    )
    def get_results(_: Auth, session_id: str):
        try:
            return manager.get_results(session_id)
        except SessionNotFoundError:
            raise _not_found(f"Session {session_id!r} not found.")
        except ValueError as exc:
            raise _bad_request(str(exc))
        except FileNotFoundError as exc:
            raise _not_found(str(exc))

    return app
