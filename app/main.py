from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api import ingestion, chat
from app.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Document RAG API",
    description="Document ingestion and conversational RAG with interview booking",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(ingestion.router)
app.include_router(chat.router)


@app.get("/health")
async def health_check() -> dict:
    return {"status": "healthy"}


@app.get("/")
async def root() -> dict:
    return {
        "service": "Document RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }
