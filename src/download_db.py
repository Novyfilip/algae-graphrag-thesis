import os
from pathlib import Path
from huggingface_hub import snapshot_download
from config import CHROMA_DIR, COLLECTION_NAME

def ensure_chromadb_exists():
    """
    Checks if the ChromaDB directory exists locally.
    If not, downloads the database from Hugging Face Datasets.
    """
    # The actual database is usually stored under CHROMA_DIR
    if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
        print(f"ChromaDB found locally at {CHROMA_DIR}.")
        return

    print("ChromaDB not found locally. Downloading from Hugging Face...")
    
    # We expect an environment variable specifying the HF dataset ID.
    # Defaulting to an example ID so the code runs, but the user must update it.
    repo_id = os.getenv("HF_DATASET_ID", "Flucius/algaebot-chromadb")
    
    # Create the directory if it doesn't exist
    os.makedirs(CHROMA_DIR, exist_ok=True)
    
    try:
        # Download the snapshot of the dataset bounding to the CHROMA_DIR
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=CHROMA_DIR,
            local_dir_use_symlinks=False
        )
        print("Successfully downloaded ChromaDB!")
    except Exception as e:
        print(f"Failed to download ChromaDB from HF Dataset {repo_id}: {e}")
        print("Warning: AlgaeBot may not work without the database!")

if __name__ == "__main__":
    ensure_chromadb_exists()
