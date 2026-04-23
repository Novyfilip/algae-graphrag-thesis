"""
config.py

Shared configuration for the entire generation pipeline.
All paths, model names, and constants live here.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chromadb"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ChromaDB collection (misnamed, actually recursive_1000)
"""                  ~ORIGINAL CONFIG~
COLLECTION_NAME = "recursive_100"  # (misnamed, actually recursive_1000)
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base" """
CHUNKS_STRATEGY = "recursive_100"  # rsc or recursive_100
CHUNK_DIR = DATA_DIR / "chunks" / "recursive_1000"
COLLECTION_NAME =  f"{CHUNKS_STRATEGY}" #f"{CHUNKS_STRATEGY}" # or f"{CHUNKS_STRATEGY}_m3"

# Models
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" #"BAAI/bge-m3" or "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"  # or "BAAI/bge-reranker-v2-m3"

# Retrieval settings
N_QUERIES = 5           # number of reformulations
TOP_K_RETRIEVAL = 5     # results per reformulation from ChromaDB
TOP_K_RERANK = 5        # chunks kept after reranking

# Local vs Cloud toggles — flip these to switch between Ollama and API
USE_LOCAL_REFORMULATION = False  # True = Ollama (gemma4:e4b), False = DeepSeek API
USE_LOCAL_GENERATION = False     # True = Ollama (gemma4:e4b), False = OpenAI API

# Local models (Ollama)
LOCAL_REFORMULATION_MODEL = "gemma4:e4b"
LOCAL_GENERATION_MODEL = "gemma4:e4b"

# Cloud models (API)
API_REFORMULATION_MODEL = "deepseek-chat"
API_GENERATION_MODEL = "gpt-5-nano"

# Neo4j settings
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"

import os
from dotenv import load_dotenv
load_dotenv()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag")
USE_GRAPH = False  # Set to False to run baseline vector purely
USE_ROUTER = False # agent decides whether to use the graph or not