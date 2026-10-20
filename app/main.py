from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="MeetMyPets Moderation POC")
app.include_router(router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
