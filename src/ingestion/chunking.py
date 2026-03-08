from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

#model load
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}  # BGE models benefit from normalization
)
#variables
PROJECT_ROOT = Path("C:/Users/filip/Desktop/Thesis/project")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks" / "rsc"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
chunker = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
# Chunking functions##########

def segment_text(text, t_max=15000):
    if len(text) <= t_max:
        return [text]
    
    mid = len(text) // 2
    #nearest comma
    split_point = text.rfind('. ', 0, mid) + 2
    if split_point < 2:
        split_point = mid
    left_segments = segment_text(text[:mid], t_max)
    right_segments = segment_text(text[mid:], t_max)
    
    return left_segments + right_segments


def recursive_chunk(segments, t_chunk=1500, d=3):
    # Initial semantic chunking
    chunks = []
    for segment in segments:
        chunks.extend(chunker.create_documents([segment]))
    
    # Refining oversized chunks
    refined = []
    for chunk in chunks:
        if len(chunk.page_content) > t_chunk:
            # Re-chunk with progressively lower thresholds
            sub_chunks = refine_chunk(chunk.page_content, t_chunk, 95 - d, d)
            refined.extend(sub_chunks)
        else:
            refined.append(chunk)
    
    return refined


def refine_chunk(text, t_chunk, threshold, d):
    # Base case: small enough or threshold bottomed out
    if len(text) <= t_chunk or threshold <= 0:
        return [text]
    
    # Re-chunk with lower threshold
    chunker.breakpoint_threshold_amount = threshold
    sub_chunks = chunker.create_documents([text])
    
    # Check each sub-chunk, recurse if still too big
    results = []
    for sc in sub_chunks:
        results.extend(refine_chunk(sc.page_content, t_chunk, threshold - d, d))
    
    return results
def merge_small_chunks(refined, embedding_model, t_merge=350):
    merged = list(refined)  # works on a copy
    
    i = 0
    while i < len(merged):
        if len(merged[i]) < t_merge:
            # Compute similarity with neighbors
            chunk_embedding = embedding_model.embed_query(merged[i])
            
            sim_prev = -1
            sim_next = -1
            
            if i > 0:
                prev_embedding = embedding_model.embed_query(merged[i - 1])
                sim_prev = sum(a * b for a, b in zip(chunk_embedding, prev_embedding))#semantic sinilarity between neighbors
            
            if i < len(merged) - 1:
                next_embedding = embedding_model.embed_query(merged[i + 1])
                sim_next = sum(a * b for a, b in zip(chunk_embedding, next_embedding))
            
            # Merge with more similar neighbor
            if sim_prev >= sim_next and i > 0:
                merged[i - 1] = merged[i - 1] + "\n\n" + merged[i]
                merged.pop(i)
            elif i < len(merged) - 1:
                merged[i + 1] = merged[i] + "\n\n" + merged[i + 1]
                merged.pop(i)
            else:
                i += 1  # no neighbors, skip
        else:
            i += 1
    
    return merged
def enforce_max_size(chunks, t_final=2500):
    final = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=t_final,
        chunk_overlap=0,
    )
    for chunk in chunks:
        if len(chunk) > t_final:
            sub_chunks = splitter.create_documents([chunk])
            final.extend([sc.page_content for sc in sub_chunks])
        else:
            final.append(chunk)
    
    return final
def rsc_chunk(text: str, embedding_model) -> list[str]:
    # Stage 1: segment if too long
    segments = segment_text(text, t_max=15000)
    
    # Stage 2: initial semantic chunking
    chunker = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    chunks = []
    for segment in segments:
        chunks.extend(chunker.create_documents([segment]))
    
    # Stage 3a: recursively split large chunks
    refined = []
    for chunk in chunks:
        if len(chunk.page_content) > 1500:
            refined.extend(refine_chunk(chunk.page_content, 1500, 92, 3))
        else:
            refined.append(chunk.page_content)
    chunks = refined
    
    # Stage 3b: merge small chunks
    chunks = merge_small_chunks(chunks, embedding_model, t_merge=350)
    
    # Stage 4: force split anything still too big
    chunks = enforce_max_size(chunks, t_final=2500)
    
    return chunks
#Execution
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
strategies = ["recursive_1000", "semantic_p95", "rsc"]

# --- Recursive (fast, minutes total) ---
print("=== Recursive Character Splitting ===")
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks" / "recursive_1000"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

for i, json_path in enumerate(sorted(PROCESSED_DIR.glob("*.json"))):
    output_path = CHUNKS_DIR / f"{json_path.stem}.json"
    if output_path.exists():
        continue
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        chunks = splitter.create_documents([doc["text"]])
        chunked = {
            "filename": doc["filename"],
            "title": doc.get("title"),
            "authors": doc.get("authors"),
            "year": doc.get("year"),
            "strategy": "recursive_character",
            "chunk_size": 1000,
            "chunk_overlap": 0,
            "num_chunks": len(chunks),
            "chunks": [
                {"chunk_id": j, "text": c.page_content, "char_length": len(c.page_content)}
                for j, c in enumerate(chunks)
            ]
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunked, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"FAILED: {json_path.name}: {e}")
    if (i + 1) % 100 == 0:
        print(f"[{i + 1}] recursive done")

print("Recursive complete!\n")

# --- Semantic (medium, hours total) ---
print("=== Semantic Chunking ===")
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks" / "semantic_p95"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

chunker = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)

for i, json_path in enumerate(sorted(PROCESSED_DIR.glob("*.json"))):
    output_path = CHUNKS_DIR / f"{json_path.stem}.json"
    if output_path.exists():
        continue
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        chunks = chunker.create_documents([doc["text"]])
        chunked = {
            "filename": doc["filename"],
            "title": doc.get("title"),
            "authors": doc.get("authors"),
            "year": doc.get("year"),
            "strategy": "semantic",
            "breakpoint_type": "percentile",
            "breakpoint_threshold": 95,
            "num_chunks": len(chunks),
            "chunks": [
                {"chunk_id": j, "text": c.page_content, "char_length": len(c.page_content)}
                for j, c in enumerate(chunks)
            ]
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunked, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"FAILED: {json_path.name}: {e}")
    if (i + 1) % 50 == 0:
        print(f"[{i + 1}] semantic done")

print("Semantic complete!\n")

# --- RSC (slow, days total) ---
print("=== RSC ===")
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks" / "rsc"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

for i, json_path in enumerate(sorted(PROCESSED_DIR.glob("*.json"))):
    output_path = CHUNKS_DIR / f"{json_path.stem}.json"
    if output_path.exists():
        continue
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        chunks = rsc_chunk(doc["text"], embedding_model)
        chunked = {
            "filename": doc["filename"],
            "title": doc.get("title"),
            "authors": doc.get("authors"),
            "year": doc.get("year"),
            "strategy": "rsc",
            "t_chunk": 1500,
            "t_merge": 350,
            "t_final": 2500,
            "delta": 3,
            "num_chunks": len(chunks),
            "chunks": [
                {"chunk_id": j, "text": c, "char_length": len(c)}
                for j, c in enumerate(chunks)
            ]
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunked, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"FAILED: {json_path.name}: {e}")
    if (i + 1) % 10 == 0:
        print(f"[{i + 1}] rsc done")

print("RSC complete!")