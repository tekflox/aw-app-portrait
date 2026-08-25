# AI Portrait architecture

- **repo**: aw-app-portrait
- **layer**: app
- **technologies**: Python, FastAPI, React, browser MediaDevices

AI Portrait is a Tier-1 managed web app. `portrait_app/routes.py` exposes the app API and serves `ui/dist`; `portrait_app/state.py` persists people, relations, non-biometric MVP descriptors, and generation history under the app's durable workspace data directory.

## Connections

- `http` → **aw-workspace** — mounted at `/api/apps/aw-app-portrait` and on the per-app subdomain.
- `http` → **agents-platform-multitenant** — gallery blocks, image upload/download, and tags.
- `http` → **OpenAI Images API** — optional reference-image generation.

## Requirements

- Motion opens a short capture window; a capture is uploaded only after MediaPipe face detection and client-side quality screening.
- All gallery tags owned by the app use `portrait:` or `person:` namespaces.
- Unknown people require human naming; low-confidence matching never invents a name.
- Generated images retain participant tags and `portrait:generated`.
- Camera pixels are not streamed continuously to the backend.
