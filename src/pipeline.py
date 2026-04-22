"""
pipeline.py

Full query-time pipeline: MultiQuery retrieval -> Reranking -> Generation,
with optional knowledge graph expansion.

Usage:
    from pipeline import setup, run_pipeline

    components = setup()
    answer, contexts, top_chunks, triplets, query = run_pipeline(
        "What is Zostera marina?", components
    )
"""

from retrieval.retrieve import (
    load_embedding_model,
    load_vectorstore,
    build_retriever,
    load_graph_driver,
    expand_from_chunks,
)
from retrieval.rerank import load_reranker, rerank
from generation.generate import get_client, build_context, generate_answer


def setup():
    """
    Load all models and return pipeline components as a dictionary.
    Call this once at startup.
    """
    print("Setting up pipeline...")

    embedding_model = load_embedding_model()
    vectorstore = load_vectorstore(embedding_model)
    retriever = build_retriever(vectorstore)
    reranker = load_reranker()
    client = get_client()
    graph_driver = load_graph_driver()

    print("Pipeline ready.\n")

    return {
        "retriever": retriever,
        "reranker": reranker,
        "client": client,
        "graph_driver": graph_driver,
    }


def run_pipeline(query, components, chat_history=None, graph=None):
    """
    Runs a single query through the full pipeline.

    Args:
        query: The user's question as a string.
        components: Dictionary returned by setup().
        chat_history: Optional chat history for context.
        graph: Optional boolean to override config.USE_GRAPH toggle.

    Returns:
        A 5-tuple of (answer, contexts_list, top_chunks, triplets, query).
        The return shape is the same whether or not graph expansion runs;
        when graph is disabled or returns no results, triplets is [].

        - answer:         str, the generated response
        - contexts_list:  list[str], plain-text chunks shown in the UI
        - top_chunks:     list[(score, Document)] from rerank
        - triplets:       list[(chunk_id, subject, predicate, object, confidence)]
                          5-tuples from expand_from_chunks; [] when graph is off
        - query:          str, the original query (echoed back for callers
                          that want to pass everything to the visualizer)
    """
    if graph is None:
        from config import USE_ROUTER, USE_GRAPH
        if USE_ROUTER:
            from retrieval.router import route_query
            graph = route_query(query)
        else:
            graph = USE_GRAPH

    retriever = components["retriever"]
    reranker = components["reranker"]
    client = components["client"]
    graph_driver = components.get("graph_driver")

    # Step 1: MultiQuery retrieval
    documents = retriever.invoke(query)

    # Step 2: Rerank
    top_chunks = rerank(query, documents, reranker)

    # Step 3: Build context with metadata headers
    context, contexts_list = build_context(top_chunks)

    # Step 3b: Graph expansion. Triplets always exists after this block,
    # even if it's empty, so the return statement never references an
    # unbound name.
    triplets = []
    if graph and graph_driver:
        entry_chunk_ids = [
            doc.metadata.get("chunk_id")
            for score, doc in top_chunks
            if "chunk_id" in doc.metadata
        ]
        triplets = expand_from_chunks(entry_chunk_ids, graph_driver)

        if triplets:
            # Triplets are 5-tuples: (chunk_id, subject, predicate, object, confidence).
            # The chunk_id is used by the visualizer for provenance but is
            # not part of the human-readable fact line shown to the LLM.
            triplet_lines = [
                f"- {subject} {predicate} {obj} (confidence: {conf:.2f})"
                for _chunk_id, subject, predicate, obj, conf in triplets
            ]
            context = context + "\n\nRelated knowledge graph facts:\n" + "\n".join(triplet_lines)
            contexts_list.append("Related knowledge graph facts:\n" + "\n".join(triplet_lines))

    # Step 4: Generate answer
    answer = generate_answer(query, context, client, chat_history)

    return answer, contexts_list, top_chunks, triplets, query


# ==========================================================================
# CLI / build harness
# ==========================================================================
if __name__ == "__main__":
    import sys

    components = setup()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        answer, contexts, top_chunks, triplets, _ = run_pipeline(query, components)
        print(answer)
    else:
        print("Welcome to Algaebot. Type 'quit' to exit\n")
        while True:
            query = input("Question: ")
            if query.lower() in ("quit", "exit", "q"):
                break
            answer, contexts, top_chunks, triplets, _ = run_pipeline(query, components)
            print(f"\n{answer}\n")