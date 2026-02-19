# PaddleOCR RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you query scanned document pages processed by PaddleOCR. Built with LangChain, ChromaDB, OpenAI, and Streamlit.

## Project Structure

```
paddle_ocr_rag/
├── data/
│   ├── markdown/        ← place your .md OCR output files here
│   └── json/            ← optional: place your .json OCR output files here
├── chroma_db/           ← auto-generated vector store (git-ignored)
├── app.py               ← Streamlit chatbot UI
├── ingest.py            ← embeds documents into ChromaDB
├── rag_chain.py         ← LangChain RAG pipeline
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Prerequisites

- Python 3.11+
- An OpenAI API key
- Docker (for containerised deployment)

---

## Option A — Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/paddle_ocr_rag.git
cd paddle_ocr_rag
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up your API key

```bash
cp .env.example .env
# Edit .env and add your real OpenAI API key
```

### 4. Add your document files

Copy your OCR-processed markdown files into `data/markdown/`:

```bash
cp /path/to/your/output/markdown/*.md data/markdown/
```

### 5. Ingest documents into ChromaDB

```bash
python ingest.py
```

Run this once. Re-run it whenever you add new pages.

### 6. Launch the chatbot

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

---

## Option B — Run with Docker

### 1. Clone and set up `.env`

```bash
git clone https://github.com/your-username/paddle_ocr_rag.git
cd paddle_ocr_rag
cp .env.example .env
# Edit .env with your OpenAI API key
```

### 2. Add your markdown files

```bash
cp /path/to/your/output/markdown/*.md data/markdown/
```

### 3. Ingest documents (first-time setup)

```bash
docker-compose run --rm rag-app python ingest.py
```

### 4. Start the chatbot

```bash
docker-compose up
```

Open your browser at **http://localhost:8501**.

To stop: `docker-compose down`

---

## Adding New Pages

1. Copy new `.md` files into `data/markdown/`
2. Re-run ingest (locally or via Docker):
   ```bash
   python ingest.py
   # or
   docker-compose run --rm rag-app python ingest.py
   ```
3. Restart the app — it will pick up the updated vector store automatically.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | ChromaDB (persistent) |
| RAG Framework | LangChain |
| UI | Streamlit |
| Container | Docker + docker-compose |
