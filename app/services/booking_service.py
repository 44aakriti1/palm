from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.database import InterviewBooking
from app.services.chat_memory import ChatMemoryService
from app.services.llm_service import LLMService


class BookingService:
    
    def __init__(self) -> None:
        self.llm = LLMService()
    
    async def extract_booking_info(
        self,
        query: str,
        session_id: str,
        chat_memory: ChatMemoryService
    ) -> Optional[Dict[str, Any]]:
        import re
        from datetime import datetime, timedelta

        # Get recent history to see what user may have said before
        history = await chat_memory.get_history(session_id, limit=10)

        # Build conversation context for the LLM - but be explicit about strict extraction
        conversation = []
        for msg in history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            conversation.append(f"{role}: {content}")
        conversation.append(f"User: {query}")
        conversation_text = "\n".join(conversation)

        booking_schema = {
            "type": "object",
            "properties": {
                "has_booking_intent": {"type": "boolean"},
                "name": {"type": "string"},
                "email": {"type": "string"},
                "date": {"type": "string"},
                "time": {"type": "string"},
                "date_confidence": {"type": "string", "enum": ["explicit", "relative", "unsure"]},
                "source_quote": {"type": "string"}  # Quote from user showing where info came from
            },
            "required": ["has_booking_intent", "date_confidence"]
        }

        system_prompt = """You are a STRICT interview booking info extractor. Extract ONLY what the user EXPLICITLY stated.

CRITICAL RULES - NEVER VIOLATE:
1. name: ONLY if user said "my name is X" or "I'm X" or "call me X" - otherwise leave EMPTY
2. email: ONLY if user provided an email address (contains @) - otherwise leave EMPTY
3. date: Extract what they said, mark if it's relative like "tomorrow"
4. time: Extract what they said (e.g., "2pm", "14:00", "morning")
5. source_quote: Copy the exact phrase from user containing the booking details
6. DO NOT make up, guess, or hallucinate ANY information
7. If user said "tomorrow" or "next Monday", set date_confidence="relative" and date to the phrase
8. If you're unsure where info came from, leave that field EMPTY

Examples of VALID extraction:
- "My name is John, email john@example.com" -> name="John", email="john@example.com", source_quote="My name is John, email john@example.com"
- "Book for tomorrow at 2pm" -> date="tomorrow", time="2pm", date_confidence="relative", name="", email=""

Examples of INVALID (don't do this):
- Making up name "Aakriti" when user never said it
- Converting "tomorrow" to "2026-12-08" - keep as "tomorrow"
- Guessing email from username
"""

        try:
            result = await self.llm.generate_structured_response(
                query=conversation_text,
                output_schema=booking_schema,
                history=[],  # Already included in conversation_text
                system_prompt=system_prompt
            )

            if not result.get("has_booking_intent"):
                return None

            # Extract what LLM found
            extracted = {
                "name": result.get("name", "").strip(),
                "email": result.get("email", "").strip(),
                "date": result.get("date", "").strip(),
                "time": result.get("time", "").strip()
            }

            # Validate email looks real (basic check)
            if extracted["email"] and "@" not in extracted["email"]:
                extracted["email"] = ""

            # Don't trust relative dates unless confidence is marked
            date_confidence = result.get("date_confidence", "unsure")
            if date_confidence == "relative":
                # Keep relative date as-is, will parse later
                pass
            elif date_confidence == "unsure" and extracted["date"]:
                # If LLM is unsure about date, treat as missing
                extracted["date"] = ""

            # Determine what's missing
            missing = []
            for field in ["name", "email", "date", "time"]:
                if not extracted.get(field):
                    missing.append(field)

            return {
                **extracted,
                "missing_info": missing
            }

        except Exception:
            return None

    def parse_relative_date(self, date_str: str) -> str:
        from datetime import datetime, timedelta

        date_str = date_str.lower().strip()
        today = datetime.now()

        # Direct ISO format check
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return date_str  # Already in correct format
        except ValueError:
            pass

        # Handle relative terms
        if date_str == "tomorrow":
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        elif date_str == "today":
            return today.strftime("%Y-%m-%d")
        elif date_str.startswith("next "):
            # Handle "next Monday", "next Tuesday", etc.
            day_names = ["monday", "tuesday", "wednesday", "thursday",
                        "friday", "saturday", "sunday"]
            target_day = date_str[5:].strip()
            if target_day in day_names:
                target_idx = day_names.index(target_day)
                days_ahead = target_idx - today.weekday()
                if days_ahead <= 0:  # Target day already passed this week
                    days_ahead += 7
                # Add another 7 for "next" week
                return (today + timedelta(days=days_ahead + 7)).strftime("%Y-%m-%d")

        # Return original if we can't parse
        return date_str

    async def save_booking(
        self,
        session_id: str,
        name: str,
        email: str,
        date: str,
        time: str,
        db: AsyncSession
    ) -> InterviewBooking:
        booking = InterviewBooking(
            session_id=session_id,
            name=name,
            email=email,
            date=date,
            time=time
        )
        db.add(booking)
        await db.commit()
        await db.refresh(booking)
        return booking
    
    async def get_bookings(
        self,
        session_id: str,
        db: AsyncSession
    ) -> list[InterviewBooking]:
        result = await db.execute(
            select(InterviewBooking).where(InterviewBooking.session_id == session_id)
        )
        return result.scalars().all()
