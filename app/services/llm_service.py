"""LLM service for Groq and Gemini."""
import json
import re
from typing import List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.core.config import get_settings

settings = get_settings()


class LLMService:
    """Service for interacting with LLMs (Groq primary, Gemini fallback)."""

    def __init__(self) -> None:
        """Initialize LLM clients."""
        self.groq: ChatGroq | None = None
        self.gemini: ChatGoogleGenerativeAI | None = None

        if settings.GROQ_API_KEY:
            self.groq = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=settings.GROQ_API_KEY,
                temperature=0.7
            )

        if settings.GOOGLE_API_KEY:
            self.gemini = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-8b-latest",
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0.7
            )

    def _format_messages(
        self,
        system_prompt: str | None,
        history: List[Dict[str, Any]],
        query: str
    ) -> List:
        """Format messages for LangChain."""
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        # Note: "system" role messages from history are intentionally skipped
        # because they are injected via system_prompt, not replayed.

        messages.append(HumanMessage(content=query))
        return messages

    async def generate_response(
        self,
        query: str,
        history: List[Dict[str, Any]] | None = None,
        system_prompt: str | None = None
    ) -> str:
        """Generate a response using the LLM.

        Args:
            query: User query
            history: Previous chat history
            system_prompt: System instruction

        Returns:
            LLM response text
        """
        history = history or []
        messages = self._format_messages(system_prompt, history, query)

        # Priority: Groq → Gemini
        if self.groq:
            response = await self.groq.ainvoke(messages)
        elif self.gemini:
            response = await self.gemini.ainvoke(messages)
        else:
            raise ValueError(
                "No LLM configured. Set GROQ_API_KEY or GOOGLE_API_KEY in .env"
            )

        return str(response.content)

    async def generate_structured_response(
        self,
        query: str,
        output_schema: Dict[str, Any],
        history: List[Dict[str, Any]] | None = None,
        system_prompt: str | None = None
    ) -> Dict[str, Any]:
        """Generate a structured JSON response.

        Args:
            query: User query
            output_schema: Expected JSON schema
            history: Previous chat history
            system_prompt: System instruction

        Returns:
            Parsed JSON response dict

        Raises:
            ValueError: If the LLM response cannot be parsed as JSON
        """
        schema_str = json.dumps(output_schema, indent=2)
        structured_prompt = (
            f"{query}\n\n"
            f"Respond ONLY with a valid JSON object following this schema:\n{schema_str}\n"
            f"Do not include any explanation, markdown, or extra text. Only the JSON object."
        )

        response = await self.generate_response(structured_prompt, history, system_prompt)

        # FIX #6: Robust JSON extraction — LLMs often add surrounding text
        # even when told not to. Strip markdown fences first, then use
        # regex to find the JSON object anywhere in the response.
        cleaned = response.strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        # Try direct parse first (fast path)
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            pass

        # Fallback: extract the first {...} block from anywhere in the text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"LLM returned text that looks like JSON but couldn't be parsed: {e}\n"
                    f"Raw response: {response[:300]}"
                )

        raise ValueError(
            f"LLM did not return a JSON object. Raw response: {response[:300]}"
        )
