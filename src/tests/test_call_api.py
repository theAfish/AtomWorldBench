import json

from fastapi.testclient import TestClient

from api.server import create_app


def _make_sample_data(data_dir):
	payload = [
		{
			"input": "data_test_input\n",
			"action_prompt": "Do something to the structure.",
			"output": "data_test_output\n",
		}
	]
	with open(data_dir / "AddAtomAction.json", "w", encoding="utf-8") as f:
		json.dump(payload, f)


def test_register_issue_key_and_access_session(monkeypatch, tmp_path):
	data_dir = tmp_path / "data"
	sessions_dir = tmp_path / "sessions"
	data_dir.mkdir()
	sessions_dir.mkdir()
	_make_sample_data(data_dir)

	monkeypatch.setenv("ATOMWORLD_API_KEY", "bootstrap-secret")
	client = TestClient(create_app(str(data_dir), str(sessions_dir)))

	register_response = client.post(
		"/auth/register",
		json={
			"username": "alice",
			"email": "alice@example.com",
			"organization": "WMML",
		},
	)
	assert register_response.status_code == 201
	assert register_response.json()["username"] == "alice"

	issue_response = client.post(
		"/auth/issue-key",
		headers={"X-API-Key": "bootstrap-secret"},
		json={"username": "alice", "note": "benchmark access"},
	)
	assert issue_response.status_code == 201
	issued_payload = issue_response.json()
	assert issued_payload["username"] == "alice"
	assert issued_payload["api_key"].startswith("awb-")

	create_session_response = client.post(
		"/sessions",
		headers={"X-API-Key": issued_payload["api_key"]},
		json={"action_name": None, "limit": 1, "repeat": 1},
	)
	assert create_session_response.status_code == 201
	assert create_session_response.json()["task_count"] == 1


def test_access_info_is_public_and_curl_friendly(monkeypatch, tmp_path):
	data_dir = tmp_path / "data"
	sessions_dir = tmp_path / "sessions"
	data_dir.mkdir()
	sessions_dir.mkdir()
	_make_sample_data(data_dir)

	monkeypatch.setenv("ATOMWORLD_API_KEY", "bootstrap-secret")
	client = TestClient(create_app(str(data_dir), str(sessions_dir)))

	response = client.get("/access-info")
	assert response.status_code == 200
	payload = response.json()
	assert payload["auth"]["server_base_url"] == "http://testserver"
	assert payload["auth"]["benchmark_header"] == "X-API-Key"
	assert payload["auth"]["registration_endpoint"]["path"] == "/auth/register"
	assert payload["workflow"][0]["request"]["path"] == "/sessions"
	assert any("curl" in note.lower() for note in payload["notes"])


def test_root_page_is_public(monkeypatch, tmp_path):
	data_dir = tmp_path / "data"
	sessions_dir = tmp_path / "sessions"
	data_dir.mkdir()
	sessions_dir.mkdir()
	_make_sample_data(data_dir)

	monkeypatch.setenv("ATOMWORLD_API_KEY", "bootstrap-secret")
	client = TestClient(create_app(str(data_dir), str(sessions_dir)))

	response = client.get("/")
	assert response.status_code == 200
	assert "AtomWorldBench" in response.text
	assert "/access-info" in response.text


def test_issue_key_requires_admin_header(monkeypatch, tmp_path):
	data_dir = tmp_path / "data"
	sessions_dir = tmp_path / "sessions"
	data_dir.mkdir()
	sessions_dir.mkdir()
	_make_sample_data(data_dir)

	monkeypatch.setenv("ATOMWORLD_API_KEY", "bootstrap-secret")
	client = TestClient(create_app(str(data_dir), str(sessions_dir)))

	client.post("/auth/register", json={"username": "alice"})

	issue_response = client.post(
		"/auth/issue-key",
		headers={"X-API-Key": "not-admin"},
		json={"username": "alice"},
	)
	assert issue_response.status_code == 401
