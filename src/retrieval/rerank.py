"""
reranking.py

Cross-encoder reranking of retrieved chunks.
Takes the broad candidate set from MultiQuery retrieval
and rescores each chunk against the original query.
"""

from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL, TOP_K_RERANK


def load_reranker():
    """Load the BGE cross-encoder reranker model."""
    return CrossEncoder(RERANKER_MODEL)


def rerank(query, documents, reranker):
    """
    Rerank retrieved documents using cross-encoder scoring.
    
    Args:
        query: The original user question (not the reformulations)
        documents: List of LangChain Document objects from MultiQuery retrieval
        reranker: A loaded CrossEncoder model
    
    Returns:
        List of (score, document) tuples, sorted by score, limited to TOP_K_RERANK
    """
    if len(documents) == 0:
        return []

    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return ranked[:TOP_K_RERANK]
