# simple_rag
how to create a simple RAG with a simple text file information and notice how the Ollama and Chroma acts together to generate response on simple questions repeatedly.
# Simple RAG (Local, with Ollama + Chroma)

A fully local Retrieval-Augmented Generation (RAG) pipeline. No API keys, no
cloud calls — everything runs on your own machine using **Ollama** for the
LLM and embedding model, and **ChromaDB** as the local vector store.

## How it works

1. A source document (`my_document.txt`) is loaded and split into chunks.
2. Each chunk is embedded (turned into a vector) using Ollama's
   `nomic-embed-text` model and stored in a local Chroma vector database.
3. When you ask a question, it's embedded the same way, and Chroma finds
   the most similar chunks via vector similarity search.
4. Those chunks + your question are passed to Ollama's `llama3.2` model,
   which generates an answer grounded in the retrieved text.

**Ollama** does the "thinking" (embedding text and generating answers).
**Chroma** does the "remembering" (storing vectors and finding the closest
matches) — it has no language understanding of its own.

## Prerequisites

- macOS (or Linux/Windows with adjustments)
- Python 3.10+
- [Ollama](https://ollama.com) installed

## Setup

### 1. Install Ollama and pull the required models

```bash
brew install ollama       # if not already installed
ollama serve               # starts the local Ollama server (skip if already running as a background service)

ollama pull llama3.2          # the LLM used for generating answers
ollama pull nomic-embed-text  # the embedding model used for retrieval
```

> If `ollama serve` says the port is already in use, Ollama is likely
> already running in the background — you can skip this step.

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install langchain langchain-community langchain-ollama langchain-chroma chromadb pypdf
```

### 4. Add a source document

Place a `my_document.txt` file in the project root (or point the script at
your own file/PDF).

## Usage

```bash
python rag.py
```

You'll be prompted to ask questions in a loop:

```
Ask a question (or 'quit'): What is RAG?

--- Answer ---
...generated answer grounded in your document...

--- Sources ---
- my_document.txt
```

Type `quit` to exit.

## Project structure

```
simple_rag/
├── my_document.txt     # source document to be indexed
├── rag.py              # main RAG pipeline script
├── chroma_db/          # persisted vector store (auto-created, gitignored)
└── README.md
```

## Configuration

| Setting | Where | Notes |
|---|---|---|
| LLM model | `ChatOllama(model="llama3.2", ...)` | Swap for any pulled Ollama model, e.g. `qwen3:8b`, `mistral:7b` |
| Embedding model | `OllamaEmbeddings(model="nomic-embed-text")` | Swap for e.g. `mxbai-embed-large` |
| Chunk size / overlap | `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` | Tune based on document type |
| Retrieved chunks (`k`) | `as_retriever(search_kwargs={"k": 3})` | How many chunks are passed to the LLM per query |
| Temperature | `ChatOllama(..., temperature=0)` | `0` = deterministic, factual answers (recommended for RAG); higher = more varied/creative output |

## Notes

- Everything runs locally — no data leaves your machine, no API costs.
- The `chroma_db/` folder persists your embeddings to disk, so you don't
  need to re-embed documents on every run unless the source content
  changes.
- Model choice is a tradeoff between quality and hardware requirements —
  see [Ollama's model library](https://ollama.com/library) for options
  suited to your machine's RAM.

## Possible next steps

- Swap in a folder of PDFs instead of a single `.txt` file
- Add a reranking step to improve retrieval quality
- Build a simple Streamlit UI instead of the terminal loop
- Experiment with hybrid search (vector + keyword) or Graph RAG for
  relationship-heavy queries