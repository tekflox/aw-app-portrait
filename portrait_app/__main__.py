from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from .routes import build_routes

SLUG = "aw-app-portrait"
DEFAULT_PORT = 9475
APP_ROOT = Path(__file__).resolve().parent.parent


def _standalone_config() -> dict:
    return {
        "gallery_base_url": os.environ.get("PORTRAIT_GALLERY_URL", ""),
        "gallery_token": os.environ.get("PORTRAIT_GALLERY_TOKEN", ""),
        "bot_slug": os.environ.get("PORTRAIT_BOT_SLUG", "portrait"),
        "capture_interval_seconds": int(os.environ.get("PORTRAIT_CAPTURE_INTERVAL", "12")),
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        "image_model": os.environ.get("PORTRAIT_IMAGE_MODEL", "gpt-image-1.5"),
    }


def build_standalone_app() -> FastAPI:
    app = FastAPI(title="AI Portrait (standalone)")
    data_dir = Path(os.environ.get("PORTRAIT_DATA_DIR", APP_ROOT / ".tmp" / "portrait-data"))
    app.mount(f"/api/apps/{SLUG}", build_routes(_standalone_config, data_dir))
    return app


app = build_standalone_app()


def main() -> None:
    uvicorn.run(app, host=os.environ.get("AW_APP_HOST", "127.0.0.1"), port=int(os.environ.get("PORT", DEFAULT_PORT)))


if __name__ == "__main__":
    main()
