# RAG Studio

> A production-ready Retrieval-Augmented Generation platform with a full web UI, FAISS vector search, multi-query retrieval, and contextual compression.

---

## Overview

RAG Studio turns your documents into an interactive knowledge base. Upload PDFs, Word docs, CSVs, or plain text files, then ask natural language questions and get grounded answers with source citations — all running locally or via OpenAI.

The system goes well beyond a basic similarity search. A multi-query expansion step rewrites your question into several variants, dramatically improving recall. A contextual compression stage then strips irrelevant content from each retrieved chunk before feeding the final context to the LLM.

---

## Architecture

```
User Query
    │
    ▼
Multi-Query Expansion  ─────  LLM generates 5 query variants
    │
    ▼
FAISS Vector Search  ──────  Cosine similarity over all variants
    │
    ▼
Contextual Compression  ────  LLM filters noise from chunks
    │
    ▼
RAG Chain  ─────────────────  LLM generates grounded answer
    │
    ▼
Response + Source Citations
```

```
rag-studio/
├── app/
│   ├── config.py                  # Pydantic-settings configuration
│   ├── core/
│   │   ├── document_processor.py  # PDF/DOCX/TXT/CSV/MD ingestion
│   │   ├── embeddings.py          # HuggingFace / Ollama / OpenAI embeddings
│   │   ├── vectorstore.py         # FAISS index management
│   │   ├── retriever.py           # Multi-query + compression retriever
│   │   ├── chain.py               # Full RAG chain with history
│   │   ├── llm.py                 # Ollama / OpenAI LLM abstraction
│   │   └── session.py             # Persistent chat session management
│   └── api/
│       ├── main.py                # FastAPI application factory
│       ├── dependencies.py        # Dependency injection
│       ├── models.py              # Pydantic request/response models
│       └── routes/
│           ├── documents.py       # Upload, list, delete documents
│           ├── query.py           # Query and similarity search
│           ├── sessions.py        # Session CRUD
│           └── health.py          # Health check
├── ui/
│   ├── Home.py                    # Streamlit home page
│   └── pages/
│       ├── 1_Documents.py         # Document upload and management
│       ├── 2_Chat.py              # Interactive Q&A chat
│       ├── 3_Settings.py          # Configuration panel
│       └── 4_Analytics.py         # Usage analytics dashboard
├── tests/
│   ├── conftest.py
│   ├── test_document_processor.py
│   ├── test_session.py
│   └── test_api.py
├── storage/                       # Local document and index storage
├── requirements.txt
├── .env.example
└── Makefile
```

---

## Features

- **Multi-format ingestion** — PDF, DOCX, TXT, CSV, Markdown
- **FAISS vector index** — fast local similarity search, no external vector DB required
- **Multi-query retrieval** — LLM expands your query into 5 variants for higher recall
- **Contextual compression** — filters irrelevant content before sending to the LLM
- **Conversation history** — sessions remember previous turns for follow-up questions
- **Dual LLM support** — Ollama (local) or OpenAI (cloud), switchable via `.env`
- **Dual embedding support** — HuggingFace sentence-transformers, Ollama, or OpenAI
- **REST API** — full FastAPI backend with OpenAPI docs at `/docs`
- **Streamlit UI** — document upload, chat, settings, and analytics in the browser
- **Persistent sessions** — chat history saved to disk per session

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/punyamodi/rag-studio.git
cd rag-studio
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env to set your LLM provider and model
```

For **Ollama** (recommended, fully local):

```bash
# Install Ollama from https://ollama.com
ollama pull mistral:7b
ollama pull nomic-embed-text
```

For **OpenAI**:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
EMBEDDING_PROVIDER=openai
```

### 4. Start the API server

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
# or
python run_api.py
```

API docs available at [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Start the UI

```bash
streamlit run ui/Home.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health/` | System health and status |
| `POST` | `/api/v1/documents/upload` | Upload and index a document |
| `GET` | `/api/v1/documents/` | List all indexed documents |
| `GET` | `/api/v1/documents/{id}` | Get document metadata |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document and its index |
| `POST` | `/api/v1/query/` | Query the knowledge base |
| `POST` | `/api/v1/query/similarity` | Raw similarity search with scores |
| `POST` | `/api/v1/sessions/` | Create a chat session |
| `GET` | `/api/v1/sessions/` | List all sessions |
| `GET` | `/api/v1/sessions/{id}` | Get session with message history |
| `DELETE` | `/api/v1/sessions/{id}` | Delete a session |
| `DELETE` | `/api/v1/sessions/{id}/history` | Clear session message history |

### Example: Upload and query

```bash
# Upload a document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@report.pdf"

# Create a session
curl -X POST http://localhost:8000/api/v1/sessions/ \
  -H "Content-Type: application/json" \
  -d '{"name": "Research Session"}'

# Query
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main findings?",
    "session_id": "<session-id>",
    "use_multi_query": true,
    "use_compression": true
  }'
```

---

## Configuration

All settings are managed through environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `OLLAMA_MODEL` | `mistral:7b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OPENAI_API_KEY` | `` | OpenAI API key |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | OpenAI model name |
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface`, `ollama`, or `openai` |
| `HF_EMBED_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_K` | `6` | Number of chunks to retrieve |
| `MULTI_QUERY_COUNT` | `5` | Query variants to generate |
| `COMPRESSION_ENABLED` | `true` | Enable contextual compression |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum upload file size |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Supported LLM Models

**Ollama (local, no API key needed):**

```bash
ollama pull mistral:7b       # fast, excellent for RAG
ollama pull llama3:8b        # Meta Llama 3
ollama pull phi3:mini        # lightweight, very fast
ollama pull gemma2:9b        # Google Gemma 2
```

**OpenAI:** `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`, `gpt-4o`

---

## Legacy Code

The original prototype script is preserved in the [`legacy`](https://github.com/punyamodi/rag-studio/tree/legacy) branch.

---

## License

MIT
