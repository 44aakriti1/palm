"""Qdrant vector store service."""
from typing import List, Dict, Any
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, PayloadSchemaType
)

from app.core.config import get_settings

settings = get_settings()


class VectorStoreService:
    """Service for managing vector storage in Qdrant."""

    def __init__(self) -> None:
        """Initialize Qdrant client (no connection attempt yet)."""
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._collection_ready: bool = False

    def _ensure_collection_once(self) -> None:
        """Lazy-init: set up the collection on first actual use."""
        if not self._collection_ready:
            self._ensure_collection()
            self._collection_ready = True

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        from app.services.embeddings import EmbeddingService

        sample_embedding = EmbeddingService.generate_single_embedding("sample text")
        vector_size = len(sample_embedding)

        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name in collection_names:
            try:
                info = self.client.get_collection(self.collection_name)
                existing_size = info.config.params.vectors.size
                if existing_size != vector_size:
                    self.client.delete_collection(self.collection_name)
                    self._create_collection(vector_size)
                else:
                    self._ensure_payload_index()
            except Exception:
                self.client.delete_collection(self.collection_name)
                self._create_collection(vector_size)
        else:
            self._create_collection(vector_size)

    def _ensure_payload_index(self) -> None:
        """Ensure payload index exists for document_id field."""
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_type=PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass  # Index may already exist

    def _create_collection(self, vector_size: int) -> None:
        """Create collection with proper configuration."""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="document_id",
            field_type=PayloadSchemaType.KEYWORD
        )

    def store_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        document_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Store chunks with embeddings in Qdrant.

        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            document_id: Parent document ID
            metadata: Additional metadata to store
        """
        self._ensure_collection_once()

        points: List[PointStruct] = []

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk,
                    "document_id": document_id,
                    "chunk_index": idx,
                    **metadata
                }
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        document_filter: str | None = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using query embedding.

        Args:
            query_embedding: Embedding of the query
            limit: Number of results to return
            document_filter: Optional document ID to filter by

        Returns:
            List of search results with text and metadata
        """
        self._ensure_collection_once()

        query_filter = None
        if document_filter:
            query_filter = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_filter))]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter
        )

        return [
            {
                "text": r.payload.get("text", ""),
                "score": r.score,
                "document_id": r.payload.get("document_id"),
                "chunk_index": r.payload.get("chunk_index"),
                "metadata": {
                    k: v for k, v in r.payload.items()
                    if k not in ["text", "document_id", "chunk_index"]
                }
            }
            for r in results
        ]

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document.

        Args:
            document_id: Document ID to delete
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            )
        )
