"""
community.py

Retrieves community-level thematic summaries from a dedicated ChromaDB collection.
Used to provide corpus-wide context for abstract/thematic queries.
"""

import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_DIR, EMBEDDING_MODEL_NAME, COMMUNITY_COLLECTION_NAME


def load_community_collection():
    """Connect to the community_summaries ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    return client.get_collection(
        name=COMMUNITY_COLLECTION_NAME,
        embedding_function=embedding_fn
    )


def retrieve_community_summaries(query, collection, n_results=3, max_distance=0.35):
    """
    Retrieve community summaries relevant to a query.
    
    Returns only summaries within the distance threshold.
    Empty list means the pipeline proceeds without community context.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    summaries = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        if dist <= max_distance:
            summaries.append(doc)

    return summaries