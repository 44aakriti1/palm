from typing import Any, Dict, List, Optional

from app.services.chat_memory import ChatMemoryService
from app.services.embeddings import EmbeddingService
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStoreService


class RAGService:

    def __init__(self) -> None:
        self.vector_store = VectorStoreService()
        self.llm = LLMService()

    async def query(
        self,
        query: str,
        session_id: str,
        chat_memory: ChatMemoryService,
        document_filter: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
       
        # 1. Retrieve conversation history
        history = await chat_memory.get_history(session_id, limit=10)

        # 2. Embed query + vector search
        query_embedding = EmbeddingService.generate_single_embedding(query)
        retrieved_chunks: List[Dict[str, Any]] = self.vector_store.search(
            query_embedding=query_embedding,
            limit=top_k,
            document_filter=document_filter
        )

        # 3. Build grounded context string
        if retrieved_chunks:
            context = "\n\n---\n\n".join(
                f"[Source {i + 1}]\n{c['text']}"
                for i, c in enumerate(retrieved_chunks)
            )
        else:
            context = "No relevant context found in the document store."

        # 4. System prompt with injected context
        system_prompt = (
            "You are a helpful assistant that answers questions based on provided context.\n"
            "Use only the context below to answer. If the answer is not in the context, say so.\n\n"
            f"Context:\n{context}"
        )

        # 5. Generate answer
        answer = await self.llm.generate_response(
            query=query,
            history=history,
            system_prompt=system_prompt
        )

        # 6. Persist this turn to Redis
        await chat_memory.add_message(session_id, "user", query)
        await chat_memory.add_message(
            session_id,
            "assistant",
            answer,
            metadata={"sources": [c["document_id"] for c in retrieved_chunks]}
        )

        return {
            "answer": answer,
            "sources": retrieved_chunks,
            "session_id": session_id
        }
