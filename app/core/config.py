"""Application configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM API Keys (Groq primary, Gemini fallback)
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    
    # Vector Database (Qdrant)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "documents"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # SQLite Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./app.db"
    
    # Embedding model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra env vars (e.g., old OPENAI_API_KEY)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
