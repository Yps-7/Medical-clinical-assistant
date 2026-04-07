# 🩺 Medical Chatbot (RAG-based Clinical Reference Assistant)
Context-aware Q&amp;A over a local medical knowledge base using FAISS vector search or Groq-hosted LLMs.

## 🧠 Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline that lets you ask medical questions grounded in a curated PDF knowledge base (e.g. an encyclopedia of medicine). Instead of hallucinating, the LLM is constrained by retrieved passages from a FAISS vector store built from your documents.

Two primary entry points:
- `connect_model.py` – CLI prototype using a Groq llama (e.g.Llama-3.1-8b).
- `medibot.py` – Streamlit chat UI using a Groq-hosted model (Llama 4 Maverick) with retrieval.

## ✨ Key Features
- FAISS vector store for fast semantic retrieval
- SentenceTransformer embeddings (`all-MiniLM-L6-v2`) – switchable to remote API mode
- Modular prompt template injection
- Caching of vector store + embeddings via Streamlit resource cache

## 🏗 Architecture
```
PDF(s) --> Text Splitter --> Embeddings --> FAISS Index (vectorstore/db_faiss)
								│
User Query --> Retriever (top-k) ---------------┘
			    │
		    Prompt Assembly
			    │
		    LLM Generation (HF or Groq)
			    │
		    Answer + Source Chunks
```
