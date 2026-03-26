"""
Centralized configuration for ArxivAssistant.
All tunable parameters live here for easy adjustment.
"""

# ─── LLM Configuration ───────────────────────────────────────────────
LLM_REPO_ID = "Qwen/Qwen2.5-7B-Instruct"
LLM_TASK = "text-generation"
LLM_MAX_NEW_TOKENS = 1024
LLM_TEMPERATURE = 0.3

# ─── Embedding Model ─────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ─── Text Splitting ──────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ─── Retriever ────────────────────────────────────────────────────────
RETRIEVER_K = 4  # Number of chunks to retrieve per query

# ─── Arxiv Loading ────────────────────────────────────────────────────
MAX_ARXIV_DOCS = 3  # Number of Arxiv papers to fetch per search
