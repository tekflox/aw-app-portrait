from __future__ import annotations

import base64
import inspect
import json
from pathlib import Path
from typing import Callable

import httpx
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .state import PortraitState

APP_ROOT = Path(__file__).resolve().parent.parent
UI_DIST = APP_ROOT / "ui" / "dist"


def prepare_portrait(content: bytes) -> tuple[bytes, dict]:
    """Validate a face server-side and return a vertical, body-aware crop."""
    frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, "The uploaded file is not a readable image")
    height, width = frame.shape[:2]
    if min(height, width) < 360:
        raise HTTPException(422, "Move closer — the image is too small")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if brightness < 35:
        raise HTTPException(422, "The room is too dark")
    if sharpness < 32:
        raise HTTPException(422, "Hold still — the image is blurred")
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(70, 70))
    if not len(faces):
        raise HTTPException(422, "No clear face was found")
    x, y, face_w, face_h = max(faces, key=lambda face: face[2] * face[3])
    if face_w * face_h < width * height * 0.012:
        raise HTTPException(422, "Move closer — the face is too small")

    # A frontal face is the stable server-side anchor. Expand well below it to
    # retain torso/full body when present, then fit a portrait-friendly 4:5 crop.
    center_x = x + face_w / 2
    crop_top = max(0, int(y - face_h * 0.9))
    crop_bottom = min(height, int(y + face_h * 7.2))
    crop_height = max(face_h * 3, crop_bottom - crop_top)
    crop_width = min(width, int(crop_height * 0.8))
    crop_width = max(crop_width, int(face_w * 3.4))
    left = max(0, min(width - crop_width, int(center_x - crop_width / 2)))
    right = min(width, left + crop_width)
    portrait = frame[crop_top:crop_bottom, left:right]
    ok, encoded = cv2.imencode(".jpg", portrait, [cv2.IMWRITE_JPEG_QUALITY, 91])
    if not ok:
        raise HTTPException(500, "Could not encode the portrait crop")
    return encoded.tobytes(), {
        "face_box": [int(x - left), int(y - crop_top), int(face_w), int(face_h)],
        "crop_box": [int(left), int(crop_top), int(right - left), int(crop_bottom - crop_top)],
        "brightness": round(brightness, 1),
        "sharpness": round(sharpness, 1),
    }


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


def build_routes(
    config_provider: Callable[[], dict] | None = None,
    data_dir: Path | None = None,
    platform_config_provider: Callable[[], dict] | None = None,
) -> FastAPI:
    config_provider = config_provider or (lambda: {})
    state = PortraitState(data_dir or APP_ROOT / ".tmp" / "portrait-data")
    app = FastAPI(title="AI Portrait")
    trusted_gallery: dict[str, str] = {}

    def cfg() -> dict:
        config = dict(config_provider() or {})
        config.setdefault("capture_interval_seconds", 12)
        config.setdefault("image_model", "gpt-image-1.5")
        return config

    async def platform_config() -> dict:
        if not platform_config_provider:
            return {}
        value = platform_config_provider()
        if inspect.isawaitable(value):
            value = await value
        return dict(value or {})

    async def gallery() -> tuple[str, str]:
        config = cfg()
        base, token = str(config.get("gallery_base_url", "")).rstrip("/"), str(config.get("gallery_token", "")).strip()
        if base and token:
            return base, token
        platform = await platform_config()
        base = str(platform.get("agents_platform_base") or base).rstrip("/")
        identity_token = str(platform.get("agents_platform_token") or "").strip()
        if not base or not identity_token:
            raise HTTPException(503, "Agents Platform Runners is not connected. Configure its trusted identity first.")
        token = trusted_gallery.get("token", "")
        if not token:
            try:
                async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                    response = await client.post(
                        f"{base}/api/admin/gallery/token",
                        headers={"Authorization": f"Bearer {identity_token}"},
                        json={"bot_slug": str(config.get("bot_slug") or "portrait"), "ttl_days": 365},
                    )
            except httpx.HTTPError as exc:
                raise HTTPException(502, f"Could not initialize the trusted gallery: {exc}") from exc
            if response.status_code >= 400:
                raise HTTPException(response.status_code, f"Could not initialize the trusted gallery: {response.text[:300]}")
            token = str(response.json().get("token") or "")
            if not token:
                raise HTTPException(502, "Agents Platform returned no gallery token")
            trusted_gallery["token"] = token
        return base, token

    async def gallery_request(method: str, path: str, **kwargs) -> httpx.Response:
        base, token = await gallery()
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
        capture_blocker = None
        try:
            await gallery()
            gallery_configured = True
        except HTTPException as exc:
            gallery_configured = False
            capture_blocker = str(exc.detail)
        return {
            "ok": True,
            "gallery_configured": gallery_configured,
            "capture_ready": gallery_configured,
            "capture_blocker": capture_blocker,
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
        for image_id in person.get("photo_ids", []):
            await tag_image(image_id, "portrait:face", "portrait:identified", f"person:{person['name']}")
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
        cropped, analysis = prepare_portrait(content)
        existing = state.collection_for(descriptor)
        if len(existing["photo_ids"]) >= 10:
            return {
                "saved": False,
                "collection_complete": True,
                "person": existing["item"] if existing["kind"] == "person" else None,
                "collection_id": existing["item"]["id"],
                "sample_count": len(existing["photo_ids"]),
                "photo_ids": existing["photo_ids"],
            }
        upload = await gallery_request(
            "POST", "/upload",
            files=[("files", (photo.filename or "portrait.jpg", cropped, "image/jpeg"))],
        )
        image_id = upload.json()["images"][0]["id"]
        collection = state.register_capture(image_id, descriptor, maximum=10)
        person, confidence = collection["person"], collection["confidence"]
        tags = ["portrait:capture", "portrait:face", "portrait:server-approved", f"portrait:collection:{collection['collection_id']}", f"portrait:sample:{collection['sample_count']}"]
        if person:
            tags.extend(["portrait:identified", f"person:{person['name']}"])
        else:
            tags.append("portrait:unknown")
        await tag_image(image_id, *tags)
        return {"saved": True, "image_id": image_id, "person": person, "confidence": confidence, "tags": tags, "analysis": analysis, **{k: collection[k] for k in ("collection_id", "sample_count", "photo_ids")}}

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
