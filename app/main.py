from fastapi import FastAPI
from app.api.routes import router
import tempfile
from app.core.config import settings

app = FastAPI(title="MeetMyPets Moderation POC")
app.include_router(router)
settings.temp_dir.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(settings.temp_dir)

@app.get("/health")
def health_check():
    return {"status": "ok"}
