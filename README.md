# MeetMyPets AI Moderation — Phase 1 POC

Local proof of concept: file validation and duplicate protection for
`POST /api/v1/moderate` (spec MMP-MOD-2026-001). Gates 1 and 2 come in later phases;
until then every valid file returns `HUMAN_REVIEW` with flag `validation_passed`.

## Setup (once)

1. `winget install --id=astral-sh.uv -e` and `setx UV_CACHE_DIR D:\uv-cache`
2. `uv python install 3.12`
3. `winget install Gyan.FFmpeg`
4. Docker Desktop (disk image location on D:)
5. In the project folder: `uv sync`, then `copy .env.example .env`

## Everyday commands

| Task | Command |
|---|---|
| Start Redis | `docker compose up -d` |
| Run the API | `uv run uvicorn app.main:app --reload` |
| Test page | `uvx --with httpx --with pandas --with pillow-heif streamlit run scripts/review_ui.py` |
| Make test files | `uv run python -m scripts.make_fixtures` |
| Run tests | `uv run pytest` |
| Inspect one file | `uv run python -m scripts.inspect_file <path> image\|video` |
| Blocklist | `uv run python -m scripts.blocklist add\|check\|remove\|count ...` |

The test page runs through `uvx` in its own environment because Streamlit requires
`pillow<13`, while the service uses Pillow 13.

## Measured timings (Phase 1, validation only)

Laptop: Intel i5-10210U, 16 GB RAM, no GPU · 3 runs per file · cache bypassed · <date>

| Type | Files | Avg server time | Avg round trip | Notes |
|---|---|---|---|---|
| Image (phone JPEG) | … | … ms | … ms | |
| Image (HEIC) | … | … ms | … ms | |
| Image (screenshot PNG) | … | … ms | … ms | |
| Video (phone MP4/MOV) | … | … ms | … ms | |
| Rejected file (any) | … | … ms | … ms | fails fast |
| Duplicate (cached) | … | … ms | … ms | answered from Redis |

## Known limitations

- CSAM check is a placeholder (decision D-13); real PhotoDNA / Thorn is needed before production.
- FastAPI receives the whole upload before the size check runs (buffered on D:, never in memory).
- If Redis is down, requests fail with HTTP 500.
- A blocklisted image that was already uploaded keeps its cached answer for up to 30 days;
  re-saved copies (new SHA-256) are caught.
