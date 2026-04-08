from enum import Enum
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from app.core.config import get_settings

settings = get_settings()


class ChunkingStrategy(str, Enum):
    recursive = "recursive"
    character = "character"


class ChunkingService:

    @staticmethod
    def chunk_text(text: str, strategy: ChunkingStrategy) -> List[str]:
        if strategy == ChunkingStrategy.recursive:
            return ChunkingService._recursive_chunk(text)
        elif strategy == ChunkingStrategy.character:
            return ChunkingService._character_chunk(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    @staticmethod
    def _recursive_chunk(text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)

    @staticmethod
    def _character_chunk(text: str) -> List[str]:
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len
        )
        return splitter.split_text(text)
