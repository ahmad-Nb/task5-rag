import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Load environment variables from the .env file
load_dotenv()

# Initialize the Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DOCS_DIR = "novatech_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "openai/gpt-oss-120b"

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Note: If Groq throws a model name error, change this to "llama3-70b-8192"
LLM_MODEL_NAME = "openai/gpt-oss-120b"

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Ground truth mapping for hit@3 evaluation
GROUND_TRUTH = {
    "How long is the warranty on a TV?": ["warranty.md"],
    "How many days do I have to return a product?": [
        "returns_policy_v2.0.md", "returns_policy_v2.md", "returns-policy-v2.md", "returns-policy.md"
    ],
    "I paid by bank transfer. How long until I get my money back after I return an item?": ["refunds.md"],
    "Do you deliver to Kuwait?": [
        "shipping_and_delivery.md", "shipping-and-delivery.md", "shipping.md"
    ],
    "Can I return earbuds I opened if they work fine?": [
        "returns_policy_v2.0.md", "returns_policy_v2.md", "returns-policy-v2.md", "returns-policy.md"
    ],
    "My order arrived broken. What should I do?": [
        "damaged_defective_items.md", "damaged-items.md", "damaged_items.md"
    ],
    "How much is delivery to Irbid for a 40 JOD order, and how long does it take?": [
        "shipping_and_delivery.md", "shipping-and-delivery.md", "shipping.md"
    ],
    "Who is the CEO of NovaTech?": [],
    "Is my laptop battery covered for 2 years?": ["warranty.md"],
    "Do you offer a student discount?": [],
    "Is the Irbid store open on Friday?": [
        "stores_and_service_center.md", "stores-and-service-center.md", "stores.md"
    ],
    "I'm a Gold member. I bought headphones 40 days ago and never opened them. Can I return them?": [
        "novapoints_loyalty_program.md", "loyalty.md", "returns_policy_v2.0.md", "returns-policy.md"
    ]
}

# ==============================================================================
# 2. CHUNKING & INDEXING PIPELINE
# ==============================================================================
def load_and_chunk(docs_dir: str, chunk_size: int, overlap: int):
    """Splits documents into overlapping character chunks while retaining file metadata."""
    chunks = []
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"Directory '{docs_dir}' not found.")

    for fname in sorted(os.listdir(docs_dir)):
        fpath = os.path.join(docs_dir, fname)
        # Accept both markdown (.md) and text (.txt) files
        if os.path.isfile(fpath) and fname.lower().endswith((".md", ".txt")):
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            start = 0
            while start < len(content):
                end = min(start + chunk_size, len(content))
                chunk_str = content[start:end]
                chunks.append({
                    "text": chunk_str,
                    "source": fname
                })
                if end == len(content):
                    break
                start += (chunk_size - overlap)
    return chunks

def build_vector_store(chunks):
    """Creates a FAISS index from document chunks."""
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# ==============================================================================
# 3. RETRIEVAL INTERFACE (Mandatory for Thursday submission)
# ==============================================================================
def search_policies(question: str, k: int = 3, chunks_ref=None, index_ref=None) -> list:
    """Retrieves top k document chunks for a question."""
    chunks_to_use = chunks_ref if chunks_ref is not None else active_chunks
    index_to_use = index_ref if index_ref is not None else active_index

    q_emb = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index_to_use.search(q_emb, k)
    
    results = []
    for idx in indices[0]:
        if idx < len(chunks_to_use):
            results.append(chunks_to_use[idx])
    return results

# ==============================================================================
# 4. RAG GENERATION
# ==============================================================================
def ask_rag(question: str, k: int = 3):
    """Retrieves relevant context and queries Groq."""
    retrieved = search_policies(question, k=k)
    
    context_blocks = []
    for i, c in enumerate(retrieved):
        context_blocks.append(f"--- Document [{i+1}]: {c['source']} ---\n{c['text']}")
    context_str = "\n\n".join(context_blocks)
    
    prompt = f"""You are a professional NovaTech customer service assistant.
Answer the customer's question strictly using ONLY the provided context below.

CRITICAL RULES:
1. If the context does not explicitly contain the information needed, state: "The provided documents do not contain information to answer this question." Do not fabricate or extrapolate.
2. If conflicting policy versions exist, prioritize Version 2.0 (effective March 2026) over older versions (v1.3 or 2025 FAQs).
3. Always state which source document(s) you derived your answer from.

Context:
{context_str}

Customer Question:
{question}
"""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=LLM_MODEL_NAME,
    )
    
    return chat_completion.choices[0].message.content.strip(), retrieved

# ==============================================================================
# 5. RETRIEVAL PERFORMANCE BENCHMARK (hit@3)
# ==============================================================================
def evaluate_hit_at_3(chunk_size: int, overlap: int):
    """Measures hit@3 retrieval rate across the test questions."""
    chunks = load_and_chunk(DOCS_DIR, chunk_size, overlap)
    index = build_vector_store(chunks)
    
    hits = 0
    evaluable_questions = 0

    for q, target_docs in GROUND_TRUTH.items():
        if not target_docs:
            continue  # Skip unanswerable queries (CEO, student discount)
        
        evaluable_questions += 1
        retrieved = search_policies(q, k=3, chunks_ref=chunks, index_ref=index)
        
        # 1. Safely extract sources and convert to lowercase
        retrieved_sources = []
        for r in retrieved:
            if isinstance(r, dict) and "source" in r:
                retrieved_sources.append(r["source"].lower())
        
        # 2. Check if any target document matches the retrieved sources
        hit = False
        for target in target_docs:
            target_clean = target.lower().replace(".md", "")
            for src in retrieved_sources:
                if target.lower() in src or target_clean in src:
                    hit = True
                    break
            if hit:
                break
                
        if hit:
            hits += 1

    hit_rate = (hits / evaluable_questions) * 100 if evaluable_questions > 0 else 0
    print(f"Chunk Size {chunk_size} (overlap {overlap}) -> hit@3: {hits}/{evaluable_questions} ({hit_rate:.1f}%)")
    return hit_rate

# ==============================================================================
# 6. EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(DOCS_DIR):
        print(f"Directory '{DOCS_DIR}' not found. Please populate it with your 14 .md files.")
        exit(1)

    print("\n--- Evaluating Retrieval Performance (hit@3) ---")
    evaluate_hit_at_3(chunk_size=500, overlap=50)
    evaluate_hit_at_3(chunk_size=250, overlap=30)

    print("\n--- Initializing Active System (Chunk Size: 500) ---")
    active_chunks = load_and_chunk(DOCS_DIR, chunk_size=500, overlap=50)
    active_index = build_vector_store(active_chunks)

    print("\n--- Running 12 Test Questions ---")
    for q in GROUND_TRUTH.keys():
        print(f"\nQ: {q}")
        ans, retrieved = ask_rag(q, k=3)
        sources = list(set([r["source"] for r in retrieved]))
        print(f"Retrieved Sources: {sources}")
        print(f"System Answer:\n{ans}")
        print("-" * 70)