from __future__ import annotations

import sys
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from portrait_app.routes import build_routes  # noqa: E402


def test_trusted_runner_identity_initializes_gallery_automatically(tmp_path, monkeypatch):
    from portrait_app import routes

    original_client = httpx.AsyncClient
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"token": "portrait-token", "created": True})

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(routes.httpx, "AsyncClient", client_factory)
    app = build_routes(
        lambda: {"gallery_base_url": "https://ap.example", "bot_slug": "portrait"},
        tmp_path,
        lambda: {"agents_platform_base": "https://ap.example", "agents_platform_token": "identity-jwt"},
    )
    response = TestClient(app).get("/api/status")
    assert response.status_code == 200
    assert response.json()["capture_ready"] is True
    assert len(calls) == 1
    assert calls[0].url.path == "/api/admin/gallery/token"
    assert calls[0].headers["authorization"] == "Bearer identity-jwt"


def test_status_reports_optional_integrations(tmp_path):
    client = TestClient(build_routes(lambda: {"capture_interval_seconds": 9}, tmp_path))
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "gallery_configured": False,
        "capture_ready": False,
        "capture_blocker": "Agents Platform Runners is not connected. Configure its trusted identity first.",
        "generation_configured": False,
        "capture_interval_seconds": 9,
        "image_model": "gpt-image-1.5",
    }


def test_people_and_relations_are_persisted(tmp_path):
    client = TestClient(build_routes(lambda: {}, tmp_path))
    ana = client.post("/api/people", json={"name": "Ana"}).json()
    leo = client.post("/api/people", json={"name": "Leo"}).json()
    relation = client.post("/api/relations", json={"source_id": ana["id"], "target_id": leo["id"], "kind": "siblings"})
    assert relation.status_code == 200
    library = client.get("/api/library").json()
    assert [person["name"] for person in library["people"]] == ["Ana", "Leo"]
    assert library["relations"][0]["kind"] == "siblings"


def test_gallery_actions_require_configuration(tmp_path):
    client = TestClient(build_routes(lambda: {}, tmp_path))
    response = client.get("/api/gallery")
    assert response.status_code == 503
