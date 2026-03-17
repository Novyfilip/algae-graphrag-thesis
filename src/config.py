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
COLLECTION_NAME = "recursive_100"

# Models
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
REFORMULATION_MODEL = "gemma3:1b"
RERANKER_MODEL = "BAAI/bge-reranker-base"
GENERATION_MODEL = "gpt-5-nano"

# Retrieval settings
N_QUERIES = 5           # number of reformulations
TOP_K_RETRIEVAL = 5     # results per reformulation from ChromaDB
TOP_K_RERANK = 5        # chunks kept after reranking
