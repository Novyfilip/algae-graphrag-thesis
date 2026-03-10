# Domain-Specific Chatbot for Algae

Master's thesis project - MSc Data Science, University of Southern Denmark (Kolding)

## Overview

Algae represent a sustainable and versatile resource with growing applications across multiple industries, including biofuel production, pharmaceuticals, nutraceuticals, fertilizers, and environmental management. This project develops a domain-specific chatbot that leverages a hybrid GraphRAG approach, combining knowledge graph-based retrieval with vector similarity search to provide accurate, real-time responses to user queries about algae-related topics.

The system integrates Neo4j for structured knowledge representation with vector embeddings for semantic search, using LangChain for orchestration. This approach leverages both symbolic reasoning and semantic understanding to serve as a valuable tool for the algae industry and research community.

## Research Questions

1. **RQ1:** How can a hybrid GraphRAG architecture be effectively designed and implemented to answer domain-specific questions about algae with high accuracy and relevance?

2. **RQ2:** Which retrieval strategies and embedding models yield the best performance for retrieving relevant information from a heterogeneous corpus of algae-related documents?

3. **RQ3:** How does the integration of knowledge graphs with vector-based retrieval impact the quality and factual grounding of generated responses compared to vector-only approaches?

## Expected Deliverables

- **Data pipeline:** Automated ingestion, processing, and indexing of multi-gigabyte corpus from diverse sources (research papers, Wikipedia, blogs, catalogs)
- **Vector database:** Searchable vector store with document embeddings and metadata
- **Knowledge graph:** Neo4j-based representation capturing entities (species, cultivation methods, applications) and their relationships
- **GraphRAG chatbot:** Functional chatbot implementing hybrid retrieval combining knowledge graph traversal with vector similarity search
- **Evaluation report:** Comparative analysis of GraphRAG vs vector-only RAG with quantitative metrics

## System Architecture

The hybrid GraphRAG architecture consists of:

1. **Indexing component:** Converts documents into vector embeddings using OpenAI embeddings, Sentence-BERT, or domain-adapted alternatives
2. **Knowledge graph construction:** Extracts entities and relationships from corpus and stores them in Neo4j
3. **Hybrid retrieval:** Combines vector similarity search with graph traversal using Cypher queries
4. **Generation component:** LLM synthesizes combined context into coherent answers

## Project Structure

```
project/
├── data/
│   ├── raw/              # Original PDFs and downloads (unmodified)
│   └── processed/        # Chunked texts and cleaned data
├── src/
│   ├── ingestion/        # PDF loading, chunking
│   ├── retrieval/        # Vector search, graph queries
│   └── generation/       # LLM calls, prompts
├── notebooks/            # Experiments and exploration
├── tests/                # Unit tests
├── outputs/              # Generated results, evaluation reports
├── .env                  # API keys (not tracked)
├── .gitignore
├── requirements.txt
└── README.md
```

## Tech Stack

- **Language:** Python
- **RAG orchestration:** LangChain / LlamaIndex
- **Vector storage:** ChromaDB / Pinecone / FAISS
- **Knowledge graph:** Neo4j (Cypher queries, GraphCypherQAChain)
- **LLMs:** OpenAI API and/or open-source alternatives
- **Entity extraction:** LLM-based pipelines
- **User interface:** Streamlit / Flask

## Evaluation Metrics
- HOPE
- **Retrieval quality:** Precision@K, Recall@K, MRR
- **Answer quality:** Faithfulness, relevance, groundedness
- **Response latency**
- **Baseline comparison:** GraphRAG vs vector-only RAG

## Timeline

| Phase | Period | Focus |
|-------|--------|-------|
| 1 | Feb - Mar | Preprocessing pipeline, knowledge graph schema design, initial corpus indexing |
| 2 | Mar - Apr | Knowledge graph construction in Neo4j, RAG implementation, embedding model comparison |
| 3 | Apr - May | GraphRAG integration, chatbot interface, evaluation, thesis writing |
| 4 | May - Jun | Final testing, documentation, thesis writing, defense preparation |

## Status

Currently evaluating the best chunking strategy and gauging pdf summary and information retrieval quality. 

## Author

**Filip Nový** - finov24@student.sdu.dk

**Supervisor:** Tariq Youssef  
**Department:** Mathematics and Computer Science, University of Southern Denmark

## References

- Lewis et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks
- Gao et al. (2023). Retrieval-augmented generation for large language models: A survey
- Karpukhin et al. (2020). Dense passage retrieval for open-domain question answering
- Hogan et al. (2021). Knowledge graphs
