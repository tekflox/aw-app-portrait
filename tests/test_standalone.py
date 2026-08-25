from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from portrait_app.__main__ import SLUG, build_standalone_app  # noqa: E402


def test_standalone_app_boots_and_mounts_api(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTRAIT_DATA_DIR", str(tmp_path))
    client = TestClient(build_standalone_app())
    response = client.get(f"/api/apps/{SLUG}/api/status")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_standalone_serves_built_react_app(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTRAIT_DATA_DIR", str(tmp_path))
    client = TestClient(build_standalone_app())
    response = client.get(f"/api/apps/{SLUG}/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
