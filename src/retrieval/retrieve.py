"""
retrieval.py

Handles embedding model loading, ChromaDB vectorstore setup,
and MultiQuery retrieval using LangChain.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    REFORMULATION_MODEL,
    N_QUERIES,
    TOP_K_RETRIEVAL,
)


def load_embedding_model():
    """Load the BGE embedding model. Must match what was used during indexing."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )


def load_vectorstore(embedding_model):
    """Load the ChromaDB vectorstore with the winning chunking strategy."""
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model
    )
    print(f"Vectorstore loaded: {vectorstore._collection.count()} chunks")
    return vectorstore


def build_retriever(vectorstore):
    """
    Build the MultiQuery retriever.
    
    Uses a local LLM (gemma3:1b) to generate N_QUERIES reformulations
    of the user's question, retrieves TOP_K_RETRIEVAL results for each,
    and returns the deduplicated union.
    """
    llm = ChatOllama(model=REFORMULATION_MODEL)

    prompt_text = f"""You are an AI assistant helping to improve information retrieval
for a scientific database about algae research.

Your task is to generate {N_QUERIES} alternative versions of the
given user question. Each alternative should approach the same
information need from a different angle or use different
terminology, so that together they can retrieve a broader set
of relevant documents.

If the original question is not in English, first translate it
to English before generating alternatives. All alternative
questions must be in English.

Provide these alternative questions separated by newlines.
Do not number them. Do not add any explanation or preamble.
Only output the alternative questions.

Original question: {{question}}"""

    multi_query_prompt = PromptTemplate(
        input_variables=["question"],
        template=prompt_text
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL}),
        llm=llm,
        prompt=multi_query_prompt,
        include_original=True
    )

    return retriever
