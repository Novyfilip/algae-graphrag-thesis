import chromadb
import json
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

# Paths
PROJECT_ROOT = Path("C:/Users/filip/Desktop/Thesis/project")
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks" / "recursive_1000"

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ChromaDB
chroma_client = chromadb.PersistentClient(path=str(DATA_DIR / "chromadb"))
collection = chroma_client.get_or_create_collection(name="recursive_100")  #was meant to be 1000, messed up up

# Load all chunks from the chunked JSONs
all_ids = []
all_texts = []
all_metadatas = []

for json_path in sorted(CHUNKS_DIR.glob("*.json")):
    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    for chunk in doc["chunks"]:
        all_ids.append(f"{doc['filename']}_chunk_{chunk['chunk_id']:03d}")
        all_texts.append(chunk["text"])
        all_metadatas.append({
            "filename": doc["filename"],
            "title": doc.get("title") or "",
            "authors": ", ".join(doc.get("authors") or []),
            "year": doc.get("year") or "",
        })

print(f"Loaded {len(all_ids)} chunks")

# Embed and add to ChromaDB in batches
BATCH_SIZE = 50

for start in range(0, len(all_ids), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(all_ids))

    embeddings = embedding_model.embed_documents(all_texts[start:end])

    collection.add(
        ids=all_ids[start:end],
        embeddings=embeddings,
        documents=all_texts[start:end],
        metadatas=all_metadatas[start:end],
    )

    if (start + BATCH_SIZE) % 500 == 0:
        print(f"[{end}/{len(all_ids)}] embedded")

print(f"Done. {collection.count()} chunks in collection.")
