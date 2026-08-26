from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Callable

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .state import PortraitState

APP_ROOT = Path(__file__).resolve().parent.parent
UI_DIST = APP_ROOT / "ui" / "dist"


class PersonIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    image_id: str | None = None
    descriptor: list[float] | None = None


class PersonPatch(BaseModel):
    name: str | None = None
    aliases: list[str] | None = None
    notes: str | None = None
    image_id: str | None = None
    descriptor: list[float] | None = None


class RelationIn(BaseModel):
    source_id: str
    target_id: str
    kind: str = Field(min_length=1, max_length=80)
    notes: str = ""


class GenerateIn(BaseModel):
    prompt: str = Field(min_length=3, max_length=2000)
    person_ids: list[str] = Field(default_factory=list, max_length=6)


def build_routes(config_provider: Callable[[], dict] | None = None, data_dir: Path | None = None) -> FastAPI:
    config_provider = config_provider or (lambda: {})
    state = PortraitState(data_dir or APP_ROOT / ".tmp" / "portrait-data")
    app = FastAPI(title="AI Portrait")

    def cfg() -> dict:
        config = dict(config_provider() or {})
        config.setdefault("capture_interval_seconds", 12)
        config.setdefault("image_model", "gpt-image-1.5")
        return config

    def gallery() -> tuple[str, str]:
        config = cfg()
        base, token = str(config.get("gallery_base_url", "")).rstrip("/"), str(config.get("gallery_token", "")).strip()
        if not base or not token:
            raise HTTPException(503, "Configure the Agents Platform gallery URL and token first")
        return base, token

    async def gallery_request(method: str, path: str, **kwargs) -> httpx.Response:
        base, token = gallery()
        url = f"{base}/api/gallery/{token}{path}"
        try:
            async with httpx.AsyncClient(timeout=90, follow_redirects=True) as client:
                response = await client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Gallery unavailable: {exc}") from exc
        if response.status_code >= 400:
            raise HTTPException(response.status_code, f"Gallery error: {response.text[:300]}")
        return response

    async def tag_image(image_id: str, *names: str) -> None:
        for name in dict.fromkeys(n.strip() for n in names if n and n.strip()):
            await gallery_request("POST", f"/image/{image_id}/tag", json={"name": name})

    @app.get("/api/status")
    async def status() -> dict:
        config = cfg()
        gallery_configured = bool(config.get("gallery_base_url") and config.get("gallery_token"))
        return {
            "ok": True,
            "gallery_configured": gallery_configured,
            "capture_ready": gallery_configured,
            "capture_blocker": None if gallery_configured else "Gallery token is missing. Open the app settings and configure it before capturing photos.",
            "generation_configured": bool(config.get("openai_api_key")),
            "capture_interval_seconds": config["capture_interval_seconds"],
            "image_model": config["image_model"],
        }

    @app.get("/api/library")
    async def library() -> dict:
        return state.snapshot()

    @app.post("/api/people")
    async def create_person(body: PersonIn) -> dict:
        person = state.create_person(body.name, body.image_id, body.descriptor)
        if body.image_id:
            await tag_image(body.image_id, "portrait:face", "portrait:identified", f"person:{person['name']}")
        return person

    @app.patch("/api/people/{person_id}")
    async def update_person(person_id: str, body: PersonPatch) -> dict:
        changes = body.model_dump(exclude_none=True, exclude={"image_id", "descriptor"})
        person = state.update_person(person_id, changes)
        if not person:
            raise HTTPException(404, "Person not found")
        if body.image_id:
            person = state.add_sample(person_id, body.image_id, body.descriptor) or person
            await tag_image(body.image_id, "portrait:face", "portrait:identified", f"person:{person['name']}")
        return person

    @app.post("/api/relations")
    async def save_relation(body: RelationIn) -> dict:
        ids = {p["id"] for p in state.snapshot()["people"]}
        if body.source_id not in ids or body.target_id not in ids:
            raise HTTPException(400, "Both people must exist")
        if body.source_id == body.target_id:
            raise HTTPException(400, "A relation needs two different people")
        return state.save_relation(body.source_id, body.target_id, body.kind, body.notes)

    @app.get("/api/gallery")
    async def list_gallery(source: str = "") -> dict:
        suffix = f"/blocks?source={source}" if source else "/blocks"
        payload = (await gallery_request("GET", suffix)).json()
        for block in payload.get("blocks", []):
            for image in block.get("images", []):
                image["url"] = f"api/media/{image['id']}"
        return payload

    @app.get("/api/media/{image_id}")
    async def media(image_id: str) -> Response:
        response = await gallery_request("GET", f"/image/{image_id}")
        return Response(response.content, media_type=response.headers.get("content-type", "image/jpeg"), headers={"Cache-Control": "private, max-age=300"})

    @app.post("/api/captures")
    async def capture(
        photo: UploadFile = File(...),
        descriptor_json: str = Form("[]"),
        quality: float = Form(0),
    ) -> dict:
        if quality < 0.42:
            raise HTTPException(422, "Portrait quality is too low")
        try:
            descriptor = [float(v) for v in json.loads(descriptor_json)][:256]
        except (ValueError, TypeError, json.JSONDecodeError):
            raise HTTPException(400, "Invalid face descriptor")
        content = await photo.read()
        if not content or len(content) > 12 * 1024 * 1024:
            raise HTTPException(400, "Photo must be between 1 byte and 12 MB")
        upload = await gallery_request(
            "POST", "/upload",
            files=[("files", (photo.filename or "portrait.jpg", content, photo.content_type or "image/jpeg"))],
        )
        image_id = upload.json()["images"][0]["id"]
        state.remember_face(image_id, descriptor)
        person, confidence = state.match(descriptor)
        tags = ["portrait:capture", "portrait:face", "portrait:reviewed"]
        if person:
            state.add_sample(person["id"], image_id, descriptor)
            tags.extend(["portrait:identified", f"person:{person['name']}"])
        else:
            tags.append("portrait:unknown")
        await tag_image(image_id, *tags)
        return {"image_id": image_id, "person": person, "confidence": confidence, "tags": tags}

    @app.post("/api/generate")
    async def generate(body: GenerateIn) -> dict:
        config = cfg()
        api_key = str(config.get("openai_api_key", "")).strip()
        if not api_key:
            raise HTTPException(503, "Configure an OpenAI API key to enable generation")
        snapshot = state.snapshot()
        people = [p for p in snapshot["people"] if p["id"] in body.person_ids]
        if not people:
            raise HTTPException(400, "Choose at least one person with a reference portrait")
        if any(not p.get("photo_ids") for p in people):
            raise HTTPException(400, "Every selected person needs a reference portrait")

        names = [p["name"] for p in people]
        related = [r for r in snapshot["relations"] if r["source_id"] in body.person_ids and r["target_id"] in body.person_ids]
        relation_text = "; ".join(r["kind"] for r in related)
        prompt = (
            "Create a warm, playful, high-quality portrait using the supplied people as identity references. "
            f"People, in reference-image order: {', '.join(names)}. "
            + (f"Their relationships include: {relation_text}. " if relation_text else "")
            + f"Scene requested by the user: {body.prompt}. Preserve recognizable facial identity, age, and key features. "
              "Make the composition suitable for a family-friendly digital photo frame; no text or watermark."
        )
        files = []
        for index, person in enumerate(people):
            image = await gallery_request("GET", f"/image/{person['photo_ids'][0]}")
            files.append(("image[]", (f"reference-{index}.jpg", image.content, image.headers.get("content-type", "image/jpeg"))))
        data = {"model": config["image_model"], "prompt": prompt, "size": "1536x1024", "quality": "high", "output_format": "jpeg"}
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post("https://api.openai.com/v1/images/edits", headers={"Authorization": f"Bearer {api_key}"}, data=data, files=files)
        if response.status_code >= 400:
            raise HTTPException(502, f"Image generation failed: {response.text[:500]}")
        result = response.json()
        encoded = result.get("data", [{}])[0].get("b64_json")
        if not encoded:
            raise HTTPException(502, "Image generation returned no image")
        generated = base64.b64decode(encoded)
        uploaded = await gallery_request("POST", "/upload", files=[("files", ("ai-portrait.jpg", generated, "image/jpeg"))])
        image_id = uploaded.json()["images"][0]["id"]
        tags = ["portrait:generated", "portrait:display-ready", *[f"person:{name}" for name in names]]
        await tag_image(image_id, *tags)
        state.log_generation({"image_id": image_id, "prompt": body.prompt, "person_ids": body.person_ids})
        return {"image_id": image_id, "url": f"api/media/{image_id}", "prompt": body.prompt, "people": names}

    if UI_DIST.is_dir():
        app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")

    return app
