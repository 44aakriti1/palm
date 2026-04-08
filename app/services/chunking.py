"""Text chunking strategies service."""
from enum import Enum
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from app.core.config import get_settings

settings = get_settings()


# FIX #5: Use str+Enum instead of Literal so FastAPI can validate it
# correctly from Form() fields (Literal is not natively handled by FastAPI forms).
class ChunkingStrategy(str, Enum):
    recursive = "recursive"
    character = "character"


class ChunkingService:
    """Service for chunking text using different strategies."""

    @staticmethod
    def chunk_text(text: str, strategy: ChunkingStrategy) -> List[str]:
        """Chunk text using specified strategy.

        Args:
            text: The text to chunk
            strategy: The chunking strategy to use

        Returns:
            List of text chunks
        """
        if strategy == ChunkingStrategy.recursive:
            return ChunkingService._recursive_chunk(text)
        elif strategy == ChunkingStrategy.character:
            return ChunkingService._character_chunk(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    @staticmethod
    def _recursive_chunk(text: str) -> List[str]:
        """Recursive character text splitter — best for structured documents."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)

    @staticmethod
    def _character_chunk(text: str) -> List[str]:
        """Simple character text splitter — best for plain text."""
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len
        )
        return splitter.split_text(text)
