"""
generation package

Full query-time pipeline: MultiQuery retrieval -> Reranking -> Generation

Usage:
    from generation import setup, run_pipeline
    
    components = setup()
    answer, contexts = run_pipeline("What is Zostera marina?", components)
"""

from retrieval.retrieve import load_embedding_model, load_vectorstore, build_retriever
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

    print("Pipeline ready.\n")

    return {
        "retriever": retriever,
        "reranker": reranker,
        "client": client,
    }


def run_pipeline(query, components, chat_history=None):
    """
    Runs a single query through the full pipeline.
    
    Args:
        query: The user's question as a string
        components: Dictionary returned by setup()
    
    Returns:
        Tuple of (answer_string, list_of_plain_text_contexts)
    """
    retriever = components["retriever"]
    reranker = components["reranker"]
    client = components["client"]

    # Step 1: MultiQuery retrieval
    documents = retriever.invoke(query)

    # Step 2: Rerank
    top_chunks = rerank(query, documents, reranker)

    # Step 3: Build context with metadata headers
    context, contexts_list = build_context(top_chunks)

    # Step 4: Generate answer
    answer = generate_answer(query, context, client, chat_history)

    return answer, contexts_list, top_chunks
#==============================================D
#FOR BUILD, RUNNING IN CONSOLE
if __name__ == "__main__":
    import sys

    components = setup()

    if len(sys.argv) > 1:
        # default query passed as argument: python -m generation "What is Zostera marina?"
        query = " ".join(sys.argv[1:])
        answer, contexts = run_pipeline(query, components)
        print(answer)
    else:
        # Interactive mode
        print("Welcome to Algaebot. Type 'quit' to exit\n")
        while True:
            query = input("Question: ")
            if query.lower() in ("quit", "exit", "q"):
                break
            answer, contexts = run_pipeline(query, components)
            print(f"\n{answer}\n")