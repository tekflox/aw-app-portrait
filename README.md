# AI Portrait (`aw-app-portrait`)

A living AI photo frame for a wall-mounted tablet. It watches for a good portrait, files approved captures in the Agents Platform Multitenant gallery, learns names from human review, and creates playful scenes using real people as identity references.

## Product flow

1. **Frame** rotates through images tagged `portrait:generated` or `portrait:display-ready`.
2. **Camera** stays active while the app is open. MediaPipe confirms a stable person looking toward the tablet, then quietly sends a candidate frame. The server independently validates the face, light, and sharpness, creates a body-aware vertical crop, and keeps up to ten samples per identity. A manual shutter remains available.
3. **People** lets an administrator name a captured face and describe relationships such as siblings, friends, parents, or teammates.
4. **Create** selects people, adds a free-form scene instruction, generates an image from their reference portraits, and files the result back into the gallery.

Gallery tags are deliberately namespaced:

- `portrait:capture`, `portrait:face`, `portrait:unknown`
- `portrait:identified`, `person:<name>`
- `portrait:generated`, `portrait:display-ready`

## Architecture and privacy

This is a Tier-1 app inherited from `aw-app-template`. The React UI is served from the app's own HTTPS subdomain, so a tablet can load it directly and use `getUserMedia`. The Python sub-app owns only metadata (people, descriptor samples, relationships, generation history); image bytes remain in the shared Agents Platform gallery.

MediaPipe presence screening and the small appearance descriptor run locally in the browser without an LLM. OpenCV performs an independent server-side face and image-quality check before anything enters the gallery. Captures are grouped into ten-sample identity collections; naming any sample adopts the whole collection. The descriptor is a lightweight MVP aid, not biometric-grade recognition. Production face recognition should replace this module with a consented, audited embedding model before unattended identification is trusted.

## Configuration

Open the app's Settings entry and set:

- `gallery_base_url` — public Agents Platform Multitenant base URL.
- `gallery_token` — active gallery share token for upload/list/tag operations.
- `capture_interval_seconds` — how often an acceptable face may be captured.
- `openai_api_key` — optional; enables reference-image generation.
- `image_model` — defaults to `gpt-image-1.5`.

The gallery token currently follows the existing gallery API's expiry policy. A durable service-token API is the preferred follow-up for an always-on appliance.

## Development

```bash
cd ui && npm install && npm run build
cd ..
python -m pytest tests -q
python tests/validate_manifest.py
PORTRAIT_GALLERY_URL=https://... PORTRAIT_GALLERY_TOKEN=... python -m portrait_app
```

Open `http://127.0.0.1:9475/api/apps/aw-app-portrait/`. Browser camera access requires localhost or HTTPS.

## Roadmap

- Durable scoped gallery service credential instead of a 24-hour share token.
- Consent/retention controls per person and a one-tap “forget me” action.
- Proper on-device face embeddings with liveness/duplicate suppression.
- Background generation recipes, quiet hours, and content-safety review.
- Presence-aware display: show relevant group memories when known people approach.
