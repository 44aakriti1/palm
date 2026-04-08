from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Literal
import uuid
from datetime import datetime

from app.db.database import get_db, DocumentMetadata
from app.services.file_extractor import FileExtractor
from app.services.chunking import ChunkingService, ChunkingStrategy
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStoreService

router = APIRouter(prefix="/ingestion", tags=["Document Ingestion"])


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunking_strategy: ChunkingStrategy = Form("recursive"),
    db: AsyncSession = Depends(get_db)
) -> dict:
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    allowed_extensions = (".pdf", ".txt")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Only {allowed_extensions} files are supported"
        )
    
    try:
        content = await file.read()
        
        text = FileExtractor.extract_text(content, file.filename)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from file")
        
        document_id = str(uuid.uuid4())
        
        chunks = ChunkingService.chunk_text(text, chunking_strategy)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from text")
        
        embeddings = EmbeddingService.generate_embeddings(chunks)
        
        vector_store = VectorStoreService()
        vector_store.store_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            metadata={
                "filename": file.filename,
                "chunking_strategy": chunking_strategy,
                "uploaded_at": datetime.utcnow().isoformat()
            }
        )
        
        file_ext = file.filename.split(".")[-1].lower()
        doc_metadata = DocumentMetadata(
            id=document_id,
            filename=file.filename,
            file_type=file_ext,
            chunk_count=len(chunks),
            chunking_strategy=chunking_strategy
        )
        db.add(doc_metadata)
        await db.commit()
        
        return {
            "status": "success",
            "document_id": document_id,
            "filename": file.filename,
            "chunking_strategy": chunking_strategy,
            "chunk_count": len(chunks),
            "message": "Document uploaded and processed successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/documents")
async def list_documents(
    db: AsyncSession = Depends(get_db)
) -> dict:
            
    from sqlalchemy import select
    result = await db.execute(select(DocumentMetadata))
    documents = result.scalars().all()
    
    return {
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "chunk_count": doc.chunk_count,
                "chunking_strategy": doc.chunking_strategy,
                "created_at": doc.created_at.isoformat() if doc.created_at else None
            }
            for doc in documents
        ]
    }
