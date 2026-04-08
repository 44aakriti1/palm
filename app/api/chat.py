"""Conversational RAG API routes."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.services.booking_service import BookingService
from app.services.chat_memory import ChatMemoryService
from app.services.rag_service import RAGService

router = APIRouter(prefix="/chat", tags=["Conversational RAG"])

# Keywords used to decide whether to run the (expensive) booking LLM call.
# FIX #7: Without this gate every single chat message made an extra LLM
# round-trip just to check for booking intent — even plain document questions.
_BOOKING_KEYWORDS: frozenset[str] = frozenset(
    {"book", "interview", "schedule", "appointment", "meeting", "slot", "reserve"}
)


# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    session_id: str
    # Note: document_id and use_openai removed - searches all docs, uses Groq/Gemini only

    class Config:
        extra = "ignore"  # Ignore old fields like document_id, use_openai


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str
    booking_info: Optional[dict] = None


# ── Service dependencies ──────────────────────────────────────────────────────
# FIX #2: Services are created via FastAPI Depends() (one instance per app
# lifetime via lru_cache-style singletons inside each service class), NOT as
# bare module-level globals.  Module-level instantiation of RAGService /
# VectorStoreService tries to connect to Qdrant at import time, crashing the
# app before startup if Qdrant isn't ready.

def get_chat_memory() -> ChatMemoryService:
    return ChatMemoryService()


def get_rag_service() -> RAGService:
    return RAGService()


def get_booking_service() -> BookingService:
    return BookingService()


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    chat_memory: ChatMemoryService = Depends(get_chat_memory),
    rag_service: RAGService = Depends(get_rag_service),
    booking_service: BookingService = Depends(get_booking_service),
) -> ChatResponse:
    """Process a conversational RAG query with chat memory.

    Args:
        request: Chat query with session ID and optional document filter
        db: Database session
        chat_memory: Redis-backed chat history service
        rag_service: Custom RAG service
        booking_service: Interview booking extraction + storage service

    Returns:
        Answer with sources and booking info if a booking was detected
    """
    try:
        # FIX #7: Only call the booking LLM if the message plausibly contains
        # booking intent.  Checking for keywords costs ~0 ms; calling the LLM
        # costs ~500–2000 ms and doubles API spend on every plain question.
        booking_info: Optional[dict] = None
        query_lower = request.query.lower()
        if any(kw in query_lower for kw in _BOOKING_KEYWORDS):
            booking_info = await booking_service.extract_booking_info(
                query=request.query,
                session_id=request.session_id,
                chat_memory=chat_memory
            )

        # Persist the booking if all required fields were extracted
        saved_booking = None
        if booking_info and not booking_info.get("missing_info"):
            if all(booking_info.get(f) for f in ("name", "email", "date", "time")):
                # Parse relative dates like "tomorrow" to actual YYYY-MM-DD
                parsed_date = booking_service.parse_relative_date(booking_info["date"])
                saved_booking = await booking_service.save_booking(
                    session_id=request.session_id,
                    name=booking_info["name"],
                    email=booking_info["email"],
                    date=parsed_date,
                    time=booking_info["time"],
                    db=db
                )
                # Update booking_info with parsed date for response
                booking_info["date"] = parsed_date

        # Handle booking flow conversationally
        response_booking_info = None
        answer = None
        sources = []

        if booking_info:
            response_booking_info = dict(booking_info)

            if saved_booking:
                # Booking is complete - confirm to user
                response_booking_info["booking_id"] = saved_booking.id
                response_booking_info["status"] = "confirmed"
                answer = (
                    f"Perfect! I've scheduled your interview for {booking_info['date']} "
                    f"at {booking_info['time']}. A confirmation will be sent to {booking_info['email']}. "
                    f"Booking ID: {saved_booking.id}"
                )
            elif booking_info.get("missing_info"):
                # Missing info - ask user for it (skip RAG, focus on booking)
                missing = ", ".join(booking_info["missing_info"])
                answer = f"I'd be happy to help you book an interview. Could you please provide: {missing}?"
                response_booking_info["status"] = "awaiting_info"

        # If no booking answer set, do normal RAG query
        if answer is None:
            # Process RAG query - searches all documents (no filter)
            result = await rag_service.query(
                query=request.query,
                session_id=request.session_id,
                chat_memory=chat_memory,
                document_filter=None  # Search across all uploaded files
            )
            answer = result["answer"]
            sources = result["sources"]

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=request.session_id,
            booking_info=response_booking_info
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.post("/clear")
async def clear_chat_history(
    session_id: str,
    chat_memory: ChatMemoryService = Depends(get_chat_memory),
) -> dict:
    """Clear chat history for a session."""
    await chat_memory.clear_history(session_id)
    return {"status": "success", "message": "Chat history cleared"}


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    chat_memory: ChatMemoryService = Depends(get_chat_memory),
) -> dict:
    """Get chat history for a session."""
    history = await chat_memory.get_history(session_id, limit=50)
    return {"session_id": session_id, "messages": history}
