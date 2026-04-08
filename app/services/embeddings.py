from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings
import numpy as np

settings = get_settings()


class EmbeddingService:
    
    _model: SentenceTransformer | None = None
    
    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            cls._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return cls._model
    
    @classmethod
    def generate_embeddings(cls, texts: List[str]) -> List[List[float]]:
        model = cls.get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    @classmethod
    def generate_single_embedding(cls, text: str) -> List[float]:
        model = cls.get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
