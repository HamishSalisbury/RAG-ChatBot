# RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in your own document corpus. Built with FastAPI, LangChain, ChromaDB, and the Anthropic Claude API.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Concepts](#key-concepts)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Large language models hallucinate when asked questions outside their training data. RAG solves this by retrieving relevant documents at query time and injecting them into the model's context window — grounding responses in real, verifiable source material.

This project implements a full RAG pipeline:

1. Documents are ingested, chunked, and embedded into a vector store
2. At query time, the top-k most semantically similar chunks are retrieved
3. Those chunks are passed as context to Claude, which generates a grounded response

Built as a personal learning project to explore the RAG pattern, vector databases, and the Anthropic API. Intended as both a functional tool and a readable reference codebase.

---

## Architecture

```
User Query
    |
    v
[Embedding Model]  -->  Query Vector
                              |
                              v
                      [ChromaDB Vector Store]
                              |
                       Top-k Chunks Retrieved
                              |
                              v
              [Prompt Template + Query + Context]
                              |
                              v
                   [Anthropic Claude (claude-sonnet-4-6)]
                              |
                              v
                      Grounded Response
```

**Ingestion pipeline (run once or on document update):**

```
Raw Documents (.pdf, .txt, .md)
    |
    v
[Document Loader]
    |
    v
[Text Splitter]  -->  Chunks (512 tokens, 50 overlap)
    |
    v
[Embedding Model]  -->  Vectors
    |
    v
[ChromaDB]  -->  Persisted Vector Store
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| API Framework | FastAPI |
| LLM Orchestration | LangChain |
| LLM Provider | Anthropic Claude (`claude-sonnet-4-6`) |
| Embeddings | `sentence-transformers` (local, free) |
| Vector Store | ChromaDB |
| Document Parsing | LangChain Document Loaders |
| Containerization | Docker + Docker Compose |
| Testing | pytest |

> **Why Claude?** Claude's 200k token context window is well-suited for RAG workloads, and the Anthropic Python SDK is straightforward to use. `claude-sonnet-4-6` balances capability and cost for a personal project.

> **Why local embeddings?** Using `sentence-transformers` for embeddings keeps costs at zero during development. The embedding model runs locally — only the LLM calls hit the Anthropic API.

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (optional, recommended)
- An Anthropic API key — get one at [console.anthropic.com](https://console.anthropic.com)

### Installation

**Option 1 — Local**

```bash
git clone https://github.com/HamishSalisbury/RAG-ChatBot.git
cd rag-chatbot

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Option 2 — Docker**

```bash
git clone https://github.com/HamishSalisbury/RAG-ChatBot.git
cd rag-chatbot
docker compose up --build
```

### Environment Variables

Copy the example env file and add your Anthropic API key:

```bash
cp .env.example .env
```

See [Configuration](#configuration) for all available variables.

### Ingest Documents

Place your source documents in the `docs/` directory, then run:

```bash
python scripts/ingest.py
```

This chunks, embeds, and stores all documents in ChromaDB. Re-run whenever your document corpus changes.

### Run the API

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Configuration

All configuration is managed via environment variables.

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Your Anthropic API key |
| `CLAUDE_MODEL` | No | `claude-sonnet-4-6` | Claude model string |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHROMA_PERSIST_DIR` | No | `./chroma_db` | Path to ChromaDB storage |
| `CHUNK_SIZE` | No | `512` | Token chunk size for splitting |
| `CHUNK_OVERLAP` | No | `50` | Token overlap between chunks |
| `TOP_K_RESULTS` | No | `4` | Number of chunks retrieved per query |
| `MAX_TOKENS` | No | `1024` | Max tokens in Claude's response |

---

## Usage

### Query via HTTP

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?"}'
```

**Response:**

```json
{
  "answer": "According to the provided documentation, refunds are available within 30 days of purchase...",
  "sources": [
    {
      "source": "docs/refund-policy.pdf",
      "page": 2,
      "chunk": "Customers are eligible for a full refund within 30 days..."
    }
  ]
}
```

### Python Client

```python
import httpx

response = httpx.post(
    "http://localhost:8000/chat",
    json={"query": "Summarize the onboarding process"}
)

data = response.json()
print(data["answer"])
print(data["sources"])
```

### Direct SDK Usage

The core chain can also be used directly without the API server:

```python
from anthropic import Anthropic
from app.retriever import retrieve_chunks
from app.chain import build_prompt

client = Anthropic()
query = "What is the return policy?"

chunks = retrieve_chunks(query, top_k=4)
prompt = build_prompt(query, chunks)

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}]
)

print(message.content[0].text)
```

---

## Project Structure

```
rag-chatbot/
├── app/
│   ├── main.py              # FastAPI app and route definitions
│   ├── chain.py             # RAG chain and prompt construction
│   ├── retriever.py         # ChromaDB retriever logic
│   └── models.py            # Pydantic request/response schemas
├── scripts/
│   └── ingest.py            # Document ingestion pipeline
├── docs/                    # Place source documents here
├── chroma_db/               # Persisted vector store (git-ignored)
├── tests/
│   ├── test_chain.py
│   └── test_ingestion.py
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Key Concepts

### Chunking Strategy

Documents are split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`. Overlap prevents context loss at chunk boundaries. A chunk size of 512 tokens balances retrieval precision against context richness.

### Embeddings

Text is converted to dense vector representations using a local `sentence-transformers` model. No API calls are made for embeddings — this keeps costs at zero during ingestion and retrieval. Semantically similar text produces vectors that are close together in high-dimensional space, enabling similarity search.

### Vector Search

ChromaDB stores embeddings and supports approximate nearest-neighbor search. At query time, the user's query is embedded and the top-k most similar document chunks are retrieved using cosine similarity.

### Prompt Construction

Retrieved chunks are injected into a structured prompt template alongside the user's query. Claude is instructed to answer only from the provided context and to clearly state when information is not available — this is the primary mechanism for reducing hallucination.

### Source Attribution

Every response includes the source document and chunk that informed the answer, enabling users to verify claims against the original material.

### Anthropic SDK

This project uses the official `anthropic` Python SDK. The SDK supports both synchronous and asynchronous usage (via `AsyncAnthropic`) and natively supports streaming responses. See [platform.claude.com/docs](https://platform.claude.com/docs) for the full API reference.

---

## What I Learned

This project was built to explore RAG fundamentals. Key takeaways:

- Chunk size and overlap have a significant effect on retrieval quality — smaller chunks retrieve more precisely but lose surrounding context
- The prompt template matters as much as retrieval — Claude needs clear instructions about when to say "I don't know"
- Local embeddings (`sentence-transformers`) are fast enough for personal projects and eliminate a second API dependency
- Source attribution is not optional — without it, a RAG system is difficult to trust or debug

---

## Roadmap

- [ ] Streaming responses via Server-Sent Events (Anthropic SDK supports this natively)
- [ ] Multi-modal support (images, tables in PDFs)
- [ ] Conversation memory for multi-turn dialogue
- [ ] Reranking layer (cross-encoder model) to improve retrieval precision
- [ ] Evaluation suite with RAGAs metrics (faithfulness, answer relevancy, context recall)
- [ ] Support for additional vector stores (Pinecone, pgvector)
- [ ] Simple Streamlit or Gradio frontend for demo purposes

---

```bash
pytest tests/
```