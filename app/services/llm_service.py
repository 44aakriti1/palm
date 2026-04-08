import json
import re
from typing import List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.core.config import get_settings

settings = get_settings()


class LLMService:

    def __init__(self) -> None:
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
        

        messages.append(HumanMessage(content=query))
        return messages

    async def generate_response(
        self,
        query: str,
        history: List[Dict[str, Any]] | None = None,
        system_prompt: str | None = None
    ) -> str:
        
        history = history or []
        messages = self._format_messages(system_prompt, history, query)

        # Priority: Groq → Gemini
        if self.groq:
            response = await self.groq.ainvoke(messages)
        elif self.gemini:
            response = await self.gemini.ainvoke(messages)
        else:
            raise ValueError("No LLM configured")

        return str(response.content)

    async def generate_structured_response(
        self,
        query: str,
        output_schema: Dict[str, Any],
        history: List[Dict[str, Any]] | None = None,
        system_prompt: str | None = None
    ) -> Dict[str, Any]:
        

        schema_str = json.dumps(output_schema, indent=2)
        structured_prompt = (
            f"{query}\n\n"
            f"Respond ONLY with a valid JSON object following this schema:\n{schema_str}\n"
            f"Do not include any explanation, markdown, or extra text. Only the JSON object."
        )

        response = await self.generate_response(structured_prompt, history, system_prompt)

        cleaned = response.strip()


        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            pass

        
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
