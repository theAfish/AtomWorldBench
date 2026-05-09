"""
AtomWorldBench REST API server.

Start with:
    atomworld serve --api-key <key> --data-folder data/simple --sessions-dir sessions

Session endpoints require the ``X-API-Key`` header. The startup key remains a
bootstrap admin credential and can also issue per-user API keys.
"""

import os
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse

from api.auth import (
    AuthPrincipal,
    AuthStore,
    InvalidApiKeyError,
    UserAlreadyExistsError,
    UserNotFoundError,
)
from api.models import (
    AccessInfoResponse,
    CreateSessionRequest,
    EvaluationResults,
    IssueApiKeyRequest,
    IssuedApiKeyResponse,
    RegisterUserRequest,
    RegisteredUserResponse,
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
    auth_store = AuthStore(root_dir=os.path.join(sessions_dir, "_auth"))

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def require_api_key(
        x_api_key: Annotated[Optional[str], Header()] = None,
    ) -> AuthPrincipal:
        expected = os.environ.get("ATOMWORLD_API_KEY", "")
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server is not configured with an API key.",
            )

        try:
            return auth_store.authenticate(x_api_key, expected)
        except InvalidApiKeyError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header.",
            )

    def require_admin_api_key(
        x_api_key: Annotated[Optional[str], Header()] = None,
    ) -> AuthPrincipal:
        expected = os.environ.get("ATOMWORLD_API_KEY", "")
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server is not configured with an admin API key.",
            )
        if x_api_key != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Admin API key required.",
            )
        return AuthPrincipal(username="admin", is_admin=True, api_key=x_api_key)

    Auth = Annotated[AuthPrincipal, Depends(require_api_key)]
    AdminAuth = Annotated[AuthPrincipal, Depends(require_admin_api_key)]

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------

    def _not_found(detail: str):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

    def _bad_request(detail: str):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    def _conflict(detail: str):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)

    # ------------------------------------------------------------------
    # Discovery endpoint
    # ------------------------------------------------------------------

    @app.get(
        "/",
        response_class=HTMLResponse,
        summary="AtomWorldBench service entry page",
    )
    def root(request: Request):
        base_url = str(request.base_url).rstrip("/")
        return f"""
<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>AtomWorldBench</title>
    <style>
      body {{ font-family: sans-serif; max-width: 900px; margin: 40px auto; padding: 0 16px; line-height: 1.5; }}
      code, pre {{ font-family: monospace; background: #f5f5f5; }}
      pre {{ padding: 12px; overflow-x: auto; border-radius: 8px; }}
      a {{ color: #0b57d0; }}
    </style>
  </head>
  <body>
    <h1>AtomWorldBench</h1>
    <p>This server is ready for direct benchmark access by users and agents.</p>
    <p><strong>Base URL:</strong> {base_url}</p>
    <ul>
      <li><a href=\"{base_url}/access-info\">Machine-readable access info</a></li>
      <li><a href=\"{base_url}/docs\">OpenAPI docs</a></li>
      <li><a href=\"{base_url}/healthz\">Health check</a></li>
    </ul>
    <p>Quick start for agents:</p>
    <pre>curl {base_url}/access-info</pre>
  </body>
</html>
"""

    @app.get(
        "/healthz",
        summary="Health check",
    )
    def healthz():
        return {"status": "ok"}

    @app.get(
        "/access-info",
        response_model=AccessInfoResponse,
        summary="Get curl-friendly access and workflow information",
    )
    def get_access_info(request: Request):
        base_url = str(request.base_url).rstrip("/")
        return AccessInfoResponse(
            api_name="AtomWorldBench API",
            version="0.1.0",
            auth={
                "server_base_url": base_url,
                "benchmark_header": "X-API-Key",
                "registration_required": True,
                "registration_endpoint": {
                    "method": "POST",
                    "path": "/auth/register",
                    "url": f"{base_url}/auth/register",
                    "authentication": "none",
                },
                "issue_key_endpoint": {
                    "method": "POST",
                    "path": "/auth/issue-key",
                    "url": f"{base_url}/auth/issue-key",
                    "authentication": "bootstrap_admin_key",
                },
                "bootstrap_key_note": (
                    "The startup --api-key value is the bootstrap admin key. "
                    "It can issue per-user keys and also works for benchmark requests."
                ),
            },
            endpoints={
                "create_session": {"method": "POST", "path": "/sessions", "url": f"{base_url}/sessions"},
                "list_tasks": {"method": "GET", "path": "/sessions/{{session_id}}/tasks", "url_template": f"{base_url}/sessions/{{session_id}}/tasks"},
                "get_task": {"method": "GET", "path": "/sessions/{{session_id}}/tasks/{{task_id}}", "url_template": f"{base_url}/sessions/{{session_id}}/tasks/{{task_id}}"},
                "submit_result": {
                    "method": "POST",
                    "path": "/sessions/{{session_id}}/tasks/{{task_id}}/submit",
                    "url_template": f"{base_url}/sessions/{{session_id}}/tasks/{{task_id}}/submit",
                },
                "evaluate": {"method": "POST", "path": "/sessions/{{session_id}}/evaluate", "url_template": f"{base_url}/sessions/{{session_id}}/evaluate"},
                "results": {"method": "GET", "path": "/sessions/{{session_id}}/results", "url_template": f"{base_url}/sessions/{{session_id}}/results"},
            },
            workflow=[
                {
                    "step": 1,
                    "name": "create_session",
                    "request": {
                        "method": "POST",
                        "path": "/sessions",
                        "url": f"{base_url}/sessions",
                        "json_body": {"action_name": None, "limit": -1, "repeat": 1},
                    },
                },
                {
                    "step": 2,
                    "name": "list_tasks",
                    "request": {
                        "method": "GET",
                        "path": "/sessions/{session_id}/tasks?offset=0&limit=500",
                        "url_template": f"{base_url}/sessions/{{session_id}}/tasks?offset=0&limit=500",
                    },
                },
                {
                    "step": 3,
                    "name": "get_task",
                    "request": {
                        "method": "GET",
                        "path": "/sessions/{session_id}/tasks/{task_id}",
                        "url_template": f"{base_url}/sessions/{{session_id}}/tasks/{{task_id}}",
                    },
                },
                {
                    "step": 4,
                    "name": "submit_result",
                    "request": {
                        "method": "POST",
                        "path": "/sessions/{session_id}/tasks/{task_id}/submit",
                        "url_template": f"{base_url}/sessions/{{session_id}}/tasks/{{task_id}}/submit",
                        "json_body": {
                            "result_cif": "<your generated CIF string>",
                            "elapsed_seconds": 1.23,
                            "token_usage": {
                                "prompt_tokens": 100,
                                "completion_tokens": 240,
                            },
                        },
                    },
                },
                {
                    "step": 5,
                    "name": "evaluate",
                    "request": {
                        "method": "POST",
                        "path": "/sessions/{session_id}/evaluate",
                        "url_template": f"{base_url}/sessions/{{session_id}}/evaluate",
                    },
                },
                {
                    "step": 6,
                    "name": "get_results",
                    "request": {
                        "method": "GET",
                        "path": "/sessions/{session_id}/results",
                        "url_template": f"{base_url}/sessions/{{session_id}}/results",
                    },
                },
            ],
            notes=[
                "Process one task at a time.",
                "Do not call /evaluate until all submissions are complete.",
                "Each task is independent and only exposes action_prompt and input_cif.",
                "This endpoint is public so agents can discover benchmark access details with curl.",
                f"Recommended public startup: atomworld serve --host 0.0.0.0 --port 50001 --api-key <ADMIN_KEY> --data-folder data/simple --sessions-dir sessions",
            ],
        )

    # ------------------------------------------------------------------
    # Registration endpoints
    # ------------------------------------------------------------------

    @app.post(
        "/auth/register",
        response_model=RegisteredUserResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Register a user for API access",
    )
    def register_user(req: RegisterUserRequest):
        try:
            return auth_store.register_user(
                username=req.username,
                email=req.email,
                organization=req.organization,
            )
        except UserAlreadyExistsError:
            raise _conflict(f"User {req.username!r} is already registered.")
        except ValueError as exc:
            raise _bad_request(str(exc))

    @app.post(
        "/auth/issue-key",
        response_model=IssuedApiKeyResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Issue an API key for a registered user",
    )
    def issue_api_key(_: AdminAuth, req: IssueApiKeyRequest):
        try:
            return auth_store.issue_api_key(username=req.username, note=req.note)
        except UserNotFoundError:
            raise _not_found(f"User {req.username!r} is not registered.")
        except ValueError as exc:
            raise _bad_request(str(exc))

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
