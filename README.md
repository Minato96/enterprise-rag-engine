# 🧠 Enterprise Document Intelligence Engine

A production-ready Retrieval-Augmented Generation (RAG) pipeline designed to ingest, process, and query complex enterprise documents (PDFs) with high precision and auditability.

---

## 🏗️ Architecture

This system is not just a script — it is a microservice designed for scalability.  
It decouples ingestion (CPU-heavy) from inference (GPU / memory-heavy) using an asynchronous API pattern.

Architecture Flow (Mermaid):

    graph LR
        A[User / Client] -->|HTTP Request| B(FastAPI Gateway)
        B -->|Query| C{RAG Chain}
        C -->|Semantic Search| D[FAISS Vector Store]
        D -->|Retrieve Top-K Chunks| C
        C -->|Augment Prompt| E[Ollama / LLM]
        E -->|Generate Answer| B
        B -->|JSON Response + Citations| A

---

## 🚀 Key Features

- 📄 **Hierarchy-Aware Ingestion**  
  Uses Unstructured.io to detect document layouts (tables, headers, sections) instead of naive text extraction.

- ⚡ **Sub-Second Retrieval**  
  FAISS with HuggingFace BGE embeddings for state-of-the-art semantic search.

- 🐋 **Dockerized Deployment**  
  Fully containerized for Dev, Staging, and Production parity.

- 🔍 **Citation-Backed Answers**  
  Strict page-level citations to minimize hallucinations and ensure auditability.

- 🛡️ **Type-Safe API**  
  FastAPI + Pydantic for strict schema validation and auto-generated docs.

---

## 🛠️ Tech Stack

| Component   | Technology                  | Reason |
|------------|-----------------------------|--------|
| LLM        | Ollama (Llama 3 / Mistral)   | Local, private, zero-cost inference |
| Embeddings | BGE-Small-EN-v1.5            | Optimized for retrieval (MTEB) |
| Vector DB  | FAISS (CPU)                  | Low-latency local search |
| Backend    | FastAPI (Async)              | High-performance ML serving |
| Container  | Docker                       | Reproducible deployments |

---

## 📂 Project Structure

    .
    ├── data/
    │   ├── raw/             # Uploaded PDFs
    │   └── vectors/         # Persisted FAISS index
    ├── src/
    │   ├── core/            # Core logic
    │   ├── ingest.py        # PDF parsing & chunking
    │   ├── vector_store.py  # Embedding pipeline
    │   ├── rag.py           # Retrieval & LLM chain
    │   └── main.py          # FastAPI entrypoint
    ├── Dockerfile
    ├── requirements.txt
    └── README.md

---

## ⚡ Getting Started

### Prerequisites

- Docker Desktop
- Ollama running locally

    ollama serve

---

### Option 1: Docker (Recommended)

Build image:

    docker build -t rag-engine .

Run container:

    docker run -p 8000:8000 rag-engine

---

### Option 2: Local Development

Create environment:

    pyenv virtualenv 3.10.12 rag-env
    pyenv local rag-env
    pip install -r requirements.txt

Run server:

    python main.py

---

## 📖 Usage Guide

Swagger UI:

    http://localhost:8000/docs

---

### 1. Upload a Document

Endpoint:

    POST /upload

Action:
- Upload a complex PDF (contracts, resumes, reports)

Result:
- Document is chunked, embedded, and indexed into FAISS

---

### 2. Chat with the Document

Endpoint:

    POST /chat

Request payload:

    {
      "query": "What are the termination conditions in the contract?"
    }

Response:

    {
      "answer": "The contract can be terminated with 30 days written notice...",
      "sources": ["contract_v1.pdf"]
    }

---

## 🔮 Future Roadmap

- [ ] Hybrid Search (BM25 + Dense Vectors)
- [ ] Cross-Encoder Re-Ranking
- [ ] Async ingestion using Celery workers