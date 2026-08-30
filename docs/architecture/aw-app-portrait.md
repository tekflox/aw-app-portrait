---
repo: architecture
path: docs/architecture/aw-app-portrait.md
source: generated
edited: false
checksum: sha256:2fded12b472e6502cecc40df29221881e736289ba72f647255ffaf00ce96eb1b
---
# AI Portrait

- **repo**: aw-app-portrait
- **layer**: app
- **technologies**: python, react
- **health** (derived): planned

Turn a tablet into a living AI photo frame: meet the people in front of its camera, organize their portraits, and place friends and family in delightful new scenes.

## Connections
- `http` → **aw-workspace** — routes mounted at /api/apps/aw-app-portrait
- `other` → **aw-app-agents-platform-runners** — Provides the shared multitenant gallery and future agent-driven generation workflows

## MCP tools
_none exposed_

## Requirements
_none documented_
