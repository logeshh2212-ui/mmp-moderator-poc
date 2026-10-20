import tempfile

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.core.logging import setup_logging
from app.validation.csam import get_csam_checker



setup_logging()
get_csam_checker()  # logs a loud warning while the placeholder is in use


# Keep FastAPI's own upload buffers on D: (rule 9).
settings.temp_dir.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(settings.temp_dir)

app = FastAPI(title="MeetMyPets Moderation POC")
app.include_router(router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
