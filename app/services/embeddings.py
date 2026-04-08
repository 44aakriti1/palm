"""Embeddings generation service."""
from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings
import numpy as np

settings = get_settings()


class EmbeddingService:
    """Service for generating embeddings using sentence-transformers."""
    
    _model: SentenceTransformer | None = None
    
    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """Get or initialize the embedding model (singleton pattern)."""
        if cls._model is None:
            cls._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return cls._model
    
    @classmethod
    def generate_embeddings(cls, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        model = cls.get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    @classmethod
    def generate_single_embedding(cls, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text string
            
        Returns:
            Embedding vector
        """
        model = cls.get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
