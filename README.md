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


   Install required dependencies:

Bash
pip install groq faiss-cpu sentence-transformers numpy python-dotenv
Configure your API Key:

Create a file named exactly .env in the root directory.

Add your Groq API key to the file:

Plaintext
GROQ_API_KEY=your_actual_api_key_here
Note: Ensure .env is listed in your .gitignore file so your key is never pushed to GitHub.

Prepare the Data:
Ensure the novatech_docs/ folder contains the 14 Markdown policy documents provided for this task.

💻 Usage
Run the main script from your terminal:

Bash
python rag.py
What the script does when executed:

Loads and chunks the 14 policy documents.

Runs a retrieval performance benchmark (hit@3) comparing 500-character vs. 250-character chunks.

Initializes the active FAISS vector store.

Iterates through 12 test questions, prints the retrieved source documents, and outputs the system's generated answer.

📁 Project Structure
rag.py: The main Python script containing the chunking, embedding, retrieval, and generation logic.

rag-notes.md: Documentation containing theoretical RAG concepts, the results table for the 12 test queries, hit@3 evaluation analysis, and debugging notes.

AI_USAGE.md: Disclosure of AI tools used for project planning, coding, and debugging.

novatech_docs/: Directory containing the 14 Markdown policy files.

.env: (Local only) Stores secure API credentials.

.gitignore: Prevents sensitive files and cache directories from being pushed to version control.
   git clone <your-repository-url>
   cd task5-rag
