"""
pipeline.py

Full query-time pipeline: MultiQuery retrieval -> Reranking -> Generation

Usage:
    from pipeline import setup, run_pipeline
    
    components = setup()
    answer, contexts, top_chunks = run_pipeline("What is Zostera marina?", components)
"""

from retrieval.retrieve import load_embedding_model, load_vectorstore, build_retriever, load_graph_driver, expand_from_chunks
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
        query: The user's question as a string
        components: Dictionary returned by setup()
        chat_history: Optional chat history for context
        graph: Optional boolean to override config.USE_GRAPH toggle
    
    Returns:
        Tuple of (answer_string, list_of_plain_text_contexts, list_of_reranked_chunks)
    """
    if graph is None:
        from config import USE_GRAPH
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

    # Step 3b: Graph Expansion if graph=True
    if graph and graph_driver:
        entry_chunk_ids = [doc.metadata.get("chunk_id") for score, doc in top_chunks if "chunk_id" in doc.metadata]
        triplets = expand_from_chunks(entry_chunk_ids, graph_driver)
        
        if triplets:
            triplet_lines = [f"- {s} {p} {o} (confidence: {c:.2f})" for s, p, o, c in triplets]
            context = context + "\n\nRelated knowledge graph facts:\n" + "\n".join(triplet_lines)
            contexts_list.append("Related knowledge graph facts:\n" + "\n".join(triplet_lines))

    # Step 4: Generate answer
    answer = generate_answer(query, context, client, chat_history)

    if graph and graph_driver and triplets:#if using graph 
        return answer, contexts_list, top_chunks, triplets
    else:
        return answer, contexts_list, top_chunks, []
#==============================================D
#FOR BUILD, RUNNING IN CONSOLE
if __name__ == "__main__":
    import sys

    components = setup()

    if len(sys.argv) > 1:
        # default query passed as argument: python -m generation "What is Zostera marina?"
        query = " ".join(sys.argv[1:])
        answer, contexts, top_chunks = run_pipeline(query, components)
        print(answer)
    else:
        # Interactive mode
        print("Welcome to Algaebot. Type 'quit' to exit\n")
        while True:
            query = input("Question: ")
            if query.lower() in ("quit", "exit", "q"):
                break
            answer, contexts, top_chunks = run_pipeline(query, components)
            print(f"\n{answer}\n")