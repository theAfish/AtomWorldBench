import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Optional


class UserAlreadyExistsError(Exception):
    pass


class UserNotFoundError(Exception):
    pass


class InvalidApiKeyError(Exception):
    pass


@dataclass(frozen=True)
class AuthPrincipal:
    username: str
    is_admin: bool
    api_key: Optional[str] = None


class AuthStore:
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self._lock = Lock()
        os.makedirs(root_dir, exist_ok=True)

    def register_user(
        self,
        username: str,
        email: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_username = username.strip()
        if not normalized_username:
            raise ValueError("username must not be empty")

        with self._lock:
            data = self._load_store()
            if normalized_username in data["users"]:
                raise UserAlreadyExistsError(normalized_username)

            created_at = datetime.now(timezone.utc).isoformat()
            user = {
                "username": normalized_username,
                "email": email,
                "organization": organization,
                "created_at": created_at,
            }
            data["users"][normalized_username] = user
            self._save_store(data)
            return user

    def issue_api_key(self, username: str, note: Optional[str] = None) -> Dict[str, Any]:
        normalized_username = username.strip()
        if not normalized_username:
            raise ValueError("username must not be empty")

        with self._lock:
            data = self._load_store()
            user = data["users"].get(normalized_username)
            if user is None:
                raise UserNotFoundError(normalized_username)

            created_at = datetime.now(timezone.utc).isoformat()
            api_key = f"awb-{secrets.token_urlsafe(24)}"
            record = {
                "api_key": api_key,
                "username": normalized_username,
                "note": note,
                "created_at": created_at,
            }
            data["keys"][api_key] = record
            self._save_store(data)

        return {
            "username": normalized_username,
            "email": user.get("email"),
            "organization": user.get("organization"),
            "api_key": api_key,
            "note": note,
            "created_at": created_at,
        }

    def authenticate(
        self,
        api_key: Optional[str],
        bootstrap_api_key: Optional[str],
    ) -> AuthPrincipal:
        if api_key and bootstrap_api_key and api_key == bootstrap_api_key:
            return AuthPrincipal(username="admin", is_admin=True, api_key=api_key)

        with self._lock:
            data = self._load_store()
            record = data["keys"].get(api_key or "")

        if record is None:
            raise InvalidApiKeyError()

        return AuthPrincipal(
            username=record["username"],
            is_admin=False,
            api_key=record["api_key"],
        )

    def _store_path(self) -> str:
        return os.path.join(self.root_dir, "auth.json")

    def _load_store(self) -> Dict[str, Dict[str, Any]]:
        path = self._store_path()
        if not os.path.exists(path):
            return {"users": {}, "keys": {}}
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        payload.setdefault("users", {})
        payload.setdefault("keys", {})
        return payload

    def _save_store(self, data: Dict[str, Dict[str, Any]]) -> None:
        path = self._store_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)