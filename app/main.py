"""FastAPI application main entry point."""
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api import ingestion, chat
from app.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    # Startup
    await init_db()
    yield
    # Shutdown


app = FastAPI(
    title="Document RAG API",
    description="Document Ingestion and Conversational RAG API with Qdrant, Redis, and Gemini/OpenAI",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(ingestion.router)
app.include_router(chat.router)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root() -> dict:
    """Root endpoint with API info."""
    return {
        "message": "Document RAG API",
        "docs": "/docs",
        "endpoints": {
            "ingestion": "/ingestion/upload",
            "chat": "/chat/query",
            "health": "/health"
        }
    }
