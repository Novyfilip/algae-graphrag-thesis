"""
config.py

Shared configuration for the entire generation pipeline.
All paths, model names, and constants live here.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path("C:/Users/filip/Desktop/Thesis/project")
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chromadb"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ChromaDB collection (misnamed, actually recursive_1000)
"""                  ~ORIGINAL CONFIG~
COLLECTION_NAME = "recursive_100"  # (misnamed, actually recursive_1000)
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base" """
CHUNKS_STRATEGY = "recursive_1000"  # change to "rsc" for second run
COLLECTION_NAME = f"{CHUNKS_STRATEGY}_m3" #"recursive_100"

# Models
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"#originally "BAAI/bge-base-en-v1.5"
REFORMULATION_MODEL = "gemma3:1b"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3" #originally "BAAI/bge-reranker-base"
GENERATION_MODEL = "gpt-5-nano"

# Retrieval settings
N_QUERIES = 5           # number of reformulations
TOP_K_RETRIEVAL = 5     # results per reformulation from ChromaDB
TOP_K_RERANK = 5        # chunks kept after reranking

# Local vs API toggle
USE_LOCAL_GENERATION = True  # True = Ollama, False = OpenAI

# Models
LOCAL_GENERATION_MODEL = "qwen3:4b"  # or llama3, mistral, granite
API_GENERATION_MODEL = "gpt-5-nano"

LOCAL_REFORMULATION_MODEL = "gemma3:1b"
API_REFORMULATION_MODEL = "gpt-4o-mini"
