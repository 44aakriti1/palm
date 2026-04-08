"""Database configuration and session management."""
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import get_settings

settings = get_settings()

# Async engine for FastAPI
async_engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()


class DocumentMetadata(Base):
    """SQLAlchemy model for document metadata."""
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    chunk_count = Column(Integer, default=0)
    chunking_strategy = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class InterviewBooking(Base):
    """SQLAlchemy model for interview bookings."""
    __tablename__ = "interview_bookings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    date = Column(String, nullable=False)
    time = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# FIX #4: Correct return type for an async generator dependency.
# Using AsyncSession directly was wrong — this function yields, so it's
# an AsyncGenerator. FastAPI's Depends() needs this typed correctly.
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
