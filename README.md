
# Agentic RAG System 

This project is a demonstration of an **Agentic Retrieval-Augmented Generation (RAG)** system that mimics human-like decision-making by combining intelligent routing, adaptive retrieval, and self-reflection.

Inspired by the article on [Agentic RAG Architectures](https://www.analyticsvidhya.com/blog/2025/01/agentic-rag-system-architectures/)

---

## What It Does

- Accepts a **natural language query** from the user.
- **Routes the query** to the appropriate domain-specific vector store (Finance, Marketing, or General) using keyword and LLM-based routing (DistilBERT).
- Detects if the query is **complex** and adapts the retrieval strategy accordingly.
- Generates an **initial answer** using a lightweight open-source LLM (`Google gemini-1.5-flash`).
- Applies a **Self-Reflective RAG** layer to critique and improve the answer before displaying it.

---

## Features

- **Agentic RAG Router** (DistilBERT + Keyword)
- **Adaptive Retrieval** (based on query complexity)
- **Self-Reflective Reasoning Layer**
- **Multi-Domain Support** (Finance, Marketing, General)

---

## Tech Stack

- **LLM**: [`Google gemini-1.5-flash`]
- **Routing Model**: `DistilBERT` (via HuggingFace Transformers)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: `FAISS` (with preloaded domain-specific stores)
- **Frameworks**: LangChain, Streamlit, HuggingFace Transformers

---

## Getting Started

1. **Install dependencies**  
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**  
```bash
streamlit run app.py
```

> Make sure the `vectorstores/` directory contains the `finance/`, `marketing/`, and `general/` FAISS stores.

---

## Project Structure

```
agentic-rag-demo/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── vectorstores/           # Prebuilt FAISS stores per domain
│   ├── finance/
│   ├── marketing/
│   └── general/
├── embed_documents.py  # FAISS vector store generator
├── generate_data.py    # Script to fetch or synthesize data
├── vectorstores/       # Saved FAISS indexes (finance, marketing, general)

```

---

## Inspired By

- [7 Agentic RAG Architectures](https://www.analyticsvidhya.com/blog/2025/01/agentic-rag-system-architectures/) 

---


## 📬 Contact

Feel free to reach out or connect with me on [LinkedIn](https://www.linkedin.com/in/saurabh-singh-0528/) 

---
