from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx

from .routes import build_routes

log = logging.getLogger("aw_apps.portrait")


class PortraitAppPlugin:
    async def activate(self, ctx) -> None:
        workspace_home = Path(os.environ.get("AW_WORKSPACE_HOME", "/opt/aw-workspace/.aw-workspace"))
        data_dir = workspace_home / "data" / "aw-app-portrait"
        data_dir.mkdir(parents=True, exist_ok=True)
        async def platform_config() -> dict:
            workspace_url = os.environ.get("AW_WORKSPACE_API_URL", "http://127.0.0.1:9030").rstrip("/")
            api_key = os.environ.get("AW_WORKSPACE_API_KEY", "")
            if not api_key:
                return {}
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                response = await client.get(
                    f"{workspace_url}/api/apps/agents-platform-runners/config",
                    headers={"X-Api-Key": api_key},
                )
            if response.status_code != 200:
                return {}
            return dict(response.json().get("config") or {})

        ctx.routes.register(build_routes(
            config_provider=lambda: dict(ctx.config or {}),
            data_dir=data_dir,
            platform_config_provider=platform_config,
        ))
        log.info("AI Portrait activated; state=%s", data_dir)

    async def on_config_saved(self, ctx) -> None:
        log.info("AI Portrait configuration updated")

    async def deactivate(self) -> None:
        log.info("AI Portrait deactivated")
