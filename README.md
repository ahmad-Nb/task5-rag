# NovaTech Customer Service RAG System

A Retrieval-Augmented Generation (RAG) pipeline built to automate customer service inquiries for NovaTech. This system ingests company policy documents, indexes them using vector embeddings, and leverages a Large Language Model (LLM) to provide accurate, context-aware answers to user queries.

## 🚀 Features

* **Document Ingestion & Chunking:** Automatically processes Markdown (`.md`) policy documents, splitting them into overlapping chunks (500 characters) while retaining file metadata.
* **Semantic Vector Search:** Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to create text embeddings and `FAISS` for fast, efficient semantic retrieval.
* **High-Speed LLM Generation:** Integrated with the Groq API (using the `openai/gpt-oss-120b` model) to generate natural language answers based strictly on retrieved context.
* **Anti-Hallucination Constraints:** Prompt-engineered to refuse to answer queries if the provided documents lack the necessary facts (e.g., "The provided documents do not contain information...").
* **Retrieval Evaluation:** Includes a built-in `hit@3` benchmarking function to measure how often the correct source documents are retrieved for a set of test questions.

## 🛠️ Tech Stack

* **Python 3**
* **Groq API** (LLM Generation)
* **FAISS** (Vector Database)
* **Sentence Transformers** (Embeddings)
* **python-dotenv** (Secure API Key Management)

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd task5-rag
