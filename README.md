# Document RAG API

A FastAPI backend with two REST APIs for document ingestion and conversational RAG.

## Features

- **Document Ingestion API**: Upload PDF/TXT files, chunk with selectable strategies, generate embeddings, store in Qdrant
- **Conversational RAG API**: Custom RAG implementation with Redis chat memory, multi-turn queries, interview booking
- **LLMs**: Gemini (primary) and OpenAI (fallback)
- **Vector Store**: Qdrant
- **Chat Memory**: Redis
- **Metadata**: SQLite with SQLAlchemy async


```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0
```


### 4. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access docs at: http://localhost:8000/docs

## API Endpoints

### Document Ingestion

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingestion/upload` | POST | Upload PDF/TXT file |
| `/ingestion/documents` | GET | List uploaded documents |

**Upload Example:**
```bash
curl -X POST "http://localhost:8000/ingestion/upload" \
  -F "file=@document.pdf" \
  -F "chunking_strategy=recursive"
```

### Conversational RAG

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/query` | POST | Send query with context |
| `/chat/history/{session_id}` | GET | Get chat history |
| `/chat/clear` | POST | Clear chat history |

**Chat Example:**
```bash
curl -X POST "http://localhost:8000/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the document say about...?",
    "session_id": "user-123",
    "document_id": "optional-doc-id"
  }'
```

**Booking Example:**
```bash
curl -X POST "http://localhost:8000/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Book an interview for John Doe, john@email.com on 2024-12-25 at 10:00 AM",
    "session_id": "user-123"
  }'
```

## Chunking Strategies

- **recursive**: Best for structured documents (uses `RecursiveCharacterTextSplitter`)
- **character**: Best for plain text (uses `CharacterTextSplitter`)
