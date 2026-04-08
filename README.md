# Document RAG API

A FastAPI backend with two REST APIs for document ingestion and conversational RAG.

## Features

- **Document Ingestion API**: Upload PDF/TXT files, chunk with selectable strategies, generate embeddings, store in Qdrant
- **Conversational RAG API**: Custom RAG implementation with Redis chat memory, multi-turn queries
- **Conversational Booking**: Interview booking via natural language (extracts name, email, date, time)
- **LLMs**: Groq (primary) and Gemini (fallback)
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
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0
```

### 3. Start Infrastructure (Docker)

```bash
# Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Redis
docker run -p 6379:6379 redis:alpine
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
    "session_id": "user-123"
  }'
```

**Conversational Booking Example:**

The system extracts booking details conversationally across multiple messages:

```bash
# First message - start booking
curl -X POST "http://localhost:8000/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want to book an interview tomorrow at 2pm",
    "session_id": "user-123"
  }'
# Response: "I'd be happy to help. Could you please provide: name, email?"

# Second message - provide details (same session)
curl -X POST "http://localhost:8000/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "My name is John Doe and my email is john@example.com",
    "session_id": "user-123"
  }'
# Response: "Perfect! I've scheduled your interview for 2024-04-09 at 2pm..."
```

## Chunking Strategies

- **recursive**: Best for structured documents (uses `RecursiveCharacterTextSplitter`)
- **character**: Best for plain text (uses `CharacterTextSplitter`)
