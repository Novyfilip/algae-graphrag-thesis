import chromadb
import json
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
import sys
sys.path.append(str(Path(__file__).parent.parent))  # goes up to src/
from config import COLLECTION_NAME, EMBEDDING_MODEL_NAME, CHUNKS_STRATEGY


# Paths
PROJECT_ROOT = Path("C:/Users/filip/Desktop/Thesis/project")
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks" / CHUNKS_STRATEGY

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ChromaDB
chroma_client = chromadb.PersistentClient(path=str(DATA_DIR / "chromadb"))
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)  #recursive 100: was meant to be 1000, messed up up
# new to resume progress
existing_ids = set(collection.get()["ids"])
print(f"Already embedded: {len(existing_ids)} chunks")
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
    
    # Filter out already embedded
    batch_ids = all_ids[start:end]
    new_indices = [i for i, id in enumerate(batch_ids) if id not in existing_ids]
    
    if not new_indices:
        continue  # skip this batch entirely
    
    batch_ids = [batch_ids[i] for i in new_indices]
    batch_texts = [all_texts[start:end][i] for i in new_indices]
    batch_metas = [all_metadatas[start:end][i] for i in new_indices]
    
    embeddings = embedding_model.embed_documents(batch_texts)
    
    collection.add(
        ids=batch_ids,
        embeddings=embeddings,
        documents=batch_texts,
        metadatas=batch_metas,
    )

    if (start + BATCH_SIZE) % 500 == 0:
        print(f"[{end}/{len(all_ids)}] embedded")

print(f"Done. {collection.count()} chunks in collection.")
