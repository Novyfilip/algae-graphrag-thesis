"""
retrieval.py

Handles embedding model loading, ChromaDB vectorstore setup,
and MultiQuery retrieval using LangChain.
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    USE_LOCAL_REFORMULATION,
    LOCAL_REFORMULATION_MODEL,
    API_REFORMULATION_MODEL,
    N_QUERIES,
    TOP_K_RETRIEVAL,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
)
from neo4j import GraphDatabase

load_dotenv()


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
    
    Generates N_QUERIES reformulations of the user's question,
    retrieves TOP_K_RETRIEVAL results for each, and returns
    the deduplicated union. LLM source (local/cloud) set in config.py.
    """
    if USE_LOCAL_REFORMULATION:
        llm = ChatOllama(model=LOCAL_REFORMULATION_MODEL)
    else:
        llm = ChatOpenAI(
            model=API_REFORMULATION_MODEL,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

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

def load_graph_driver():
    """Initializes and returns the Neo4j driver instance using config variables."""
    try:
        auth = (NEO4J_USER, NEO4J_PASSWORD)
        driver = GraphDatabase.driver(NEO4J_URI, auth=auth)
        driver.verify_connectivity()
        print("GraphDatabase driver loaded successfully.")
        return driver
    except Exception as e:
        print(f"Failed to connect to Neo4j Graph Database: {e}")
        return None

def expand_from_chunks(chunk_ids, driver, max_triplets=40):
    """
    Query the connected Neo4j knowledge graph using vector chunk IDs.
    Returns 1-hop connected facts (triplets), dynamically penalizing massive hubs.
    """
    if not driver:
        print("Warning: Graph driver not available. Skipping graph expansion.")
        return []

    cypher = """
    MATCH (c:Chunk) WHERE c.chunk_id IN $chunk_ids
    MATCH (c)-[:MENTIONS]->(entity)
    
    // Degree penalty graph hub filter
    WITH entity
    WHERE size([(entity)-[]-() | 1]) < 100
    
    MATCH (entity)-[r]->(neighbor)
    WHERE type(r) IN ['FOUND_IN','PRODUCES','STUDIED_WITH',
                      'IDENTIFIED_BY','BELONGS_TO','AFFECTS','CONTAINS']
      AND r.confidence >= 0.7
    RETURN DISTINCT
        entity.name  AS subject,
        type(r)      AS predicate,
        neighbor.name AS object,
        r.confidence AS confidence
    ORDER BY r.confidence DESC
    LIMIT $max_triplets
    """
    
    with driver.session() as session:
        result = session.run(cypher, chunk_ids=chunk_ids, max_triplets=max_triplets)
        return [(r["subject"], r["predicate"], r["object"], r["confidence"]) for r in result]
