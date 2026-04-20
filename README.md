# 🌿 AlgaeBot: Domain-Specific GraphRAG

> **[🚀 Try the live demo here!](https://algaebot.filipnovy.dk)** *(If the custom URL is pending, try the DO App Platform link)*

AlgaeBot is a deeply domain-specific retrieval-augmented generation (RAG) system built to answer complex questions about algae research, cultivation, and applications. It operates over a highly technical corpus of 879 research papers from the journal *Algae*.

This project demonstrates a production-ready **Hybrid GraphRAG** approach: it combines the semantic recall of vector similarity search (ChromaDB) with the precise structural traversal of a knowledge graph (Neo4j).

---

## 🔮 How It Works

1. **Vector-Based Anchoring**: Given a user query, the system reformulates the question and performs a dense vector search across the corpus chunks using BAAI embeddings to find the most semantically relevant context.
2. **Knowledge Graph Expansion**: The system extracts the identified chunks and traverses a Neo4j knowledge graph to find 1-hop related entities and scientific facts that might have been textually disconnected but logically linked.
3. **Cross-Encoder Reranking**: All gathered context chunks are re-ranked using a cross-encoder to ensure the most relevant material sits at the top of the context window.
4. **Grounded Generation**: Finally, an LLM (such as GPT-4o or DeepSeek) synthesizes the vector chunks and the graph triplets into a highly accurate, grounded answer.

## 📊 RAGAS Evaluation Results

To ensure academic rigor, AlgaeBot was formally evaluated using the **RAGAS** framework across 51 hand-crafted questions (measuring multi-hop reasoning capability against single-hop facts).

The baseline vector-only pipeline was compared against the hybrid triplet-enriched pipeline using an independent frontier LLM as the evaluator. The metrics demonstrate that the hybrid approach provides better groundedness and context precision.

*(Note: Final ANOVA significance scores are pending analysis in the thesis).*

## 🛠️ Architecture & Tech Stack

- **Orchestration**: `LangChain`
- **Vector Database**: `ChromaDB` (Locally cached on startup from HuggingFace Datasets to bypass GitHub limits)
- **Knowledge Graph**: `Neo4j` and `neo4j-graphrag`
- **Embeddings**: `sentence-transformers` (BAAI/bge-base-en-v1.5)
- **Reranker**: `BAAI/bge-reranker-base` 
- **LLMs**: `OpenAI API` / `DeepSeek API` (with local fallbacks via `Ollama`)
- **Frontend / Deployment**: `Streamlit`, deployed via `DigitalOcean App Platform` Container.

---

## 💻 Running it Locally

If you want to spin this up yourself:

1. Clone the repository.
2. Install the cleanly pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys.
4. Run the Streamlit application:
   ```bash
   streamlit run src/algaebot.py
   ```
*(Note: Upon your first run, the system will automatically download the 1.7GB vector database snapshot from Hugging Face Datasets so you don't have to embed 879 scientific papers from scratch.)*

---

> **University of Southern Denmark (SDU)**  
> This application serves as the implementation component of my MSc Data Science thesis, supervised by Tariq Youssef.  
> **Author**: Filip Nový (finov24@student.sdu.dk)
