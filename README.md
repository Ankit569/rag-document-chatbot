
# 📄 DocChat — RAG Document Chatbot

An AI-powered document Q&A chatbot built with **Python**, **LangChain**, **Mistral AI**, and **Streamlit**.  
Upload any PDF and ask natural language questions — answers are grounded in the document using **Retrieval-Augmented Generation (RAG)**.

---

## What is RAG?

RAG (Retrieval-Augmented Generation) solves a key problem with LLMs: they don't know the contents of *your* documents.

Instead of sending the whole PDF to the LLM (expensive and slow), RAG:
1. Splits the document into small chunks
2. Converts each chunk into a vector (embedding)
3. At query time, finds the most relevant chunks
4. Sends only those chunks + your question to the LLM

This gives accurate, grounded answers with source references.

---

## Features

- Upload any PDF document
- Automatic text chunking and vector indexing with FAISS
- Semantic search to find relevant passages
- Answers grounded in document content (not hallucinated)
- Shows source page and snippet for every answer
- Persistent index — no re-processing on refresh
- Clean chat interface built with Streamlit

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| LLM | Mistral AI (`mistral-small-latest`) |
| Embeddings | Mistral AI (`mistral-embed`) |
| Vector Store | FAISS |
| Orchestration | LangChain |
| PDF Parsing | PyPDF |

---

## Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot
cd rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Mistral API key
```bash
cp .env.example .env
# Edit .env and add your key from https://console.mistral.ai
```

### 4. Run
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
rag-chatbot/
├── app.py            # Streamlit UI
├── rag.py            # RAG pipeline logic
├── requirements.txt
├── .env.example
└── README.md
```

---

## Built by

Ankit Mayur — [LinkedIn](https://linkedin.com/in/ankit-mayur) | BCA Graduate, Rani Channamma University


