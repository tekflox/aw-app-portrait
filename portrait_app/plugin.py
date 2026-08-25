from __future__ import annotations

import logging
import os
from pathlib import Path

from .routes import build_routes

log = logging.getLogger("aw_apps.portrait")


class PortraitAppPlugin:
    async def activate(self, ctx) -> None:
        workspace_home = Path(os.environ.get("AW_WORKSPACE_HOME", "/opt/aw-workspace/.aw-workspace"))
        data_dir = workspace_home / "data" / "aw-app-portrait"
        data_dir.mkdir(parents=True, exist_ok=True)
        ctx.routes.register(build_routes(config_provider=lambda: dict(ctx.config or {}), data_dir=data_dir))
        log.info("AI Portrait activated; state=%s", data_dir)

    async def on_config_saved(self, ctx) -> None:
        log.info("AI Portrait configuration updated")

    async def deactivate(self) -> None:
        log.info("AI Portrait deactivated")
