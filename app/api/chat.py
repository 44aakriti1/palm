from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.services.booking_service import BookingService
from app.services.chat_memory import ChatMemoryService
from app.services.rag_service import RAGService

router = APIRouter(prefix="/chat", tags=["Conversational RAG"])

_BOOKING_KEYWORDS: frozenset[str] = frozenset(
    {"book", "interview", "schedule", "appointment", "meeting", "slot", "reserve"}
)


class ChatRequest(BaseModel):
    query: str
    session_id: str

    class Config:
        extra = "ignore"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str
    booking_info: Optional[dict] = None

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
    try:
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
                parsed_date = booking_service.parse_relative_date(booking_info["date"])
                saved_booking = await booking_service.save_booking(
                    session_id=request.session_id,
                    name=booking_info["name"],
                    email=booking_info["email"],
                    date=parsed_date,
                    time=booking_info["time"],
                    db=db
                )
                booking_info["date"] = parsed_date

        response_booking_info = None
        answer = None
        sources = []

        if booking_info:
            response_booking_info = dict(booking_info)

            if saved_booking:
                response_booking_info["booking_id"] = saved_booking.id
                response_booking_info["status"] = "confirmed"
                answer = (
                    f"Perfect! I've scheduled your interview for {booking_info['date']} "
                    f"at {booking_info['time']}. A confirmation will be sent to {booking_info['email']}. "
                    f"Booking ID: {saved_booking.id}"
                )
            elif booking_info.get("missing_info"):
                missing = ", ".join(booking_info["missing_info"])
                answer = f"I'd be happy to help you book an interview. Could you please provide: {missing}?"
                response_booking_info["status"] = "awaiting_info"

        if answer is None:
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
    await chat_memory.clear_history(session_id)
    return {"status": "success", "message": "Chat history cleared"}


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    chat_memory: ChatMemoryService = Depends(get_chat_memory),
) -> dict:
    history = await chat_memory.get_history(session_id, limit=50)
    return {"session_id": session_id, "messages": history}
