import json
import gzip
import numpy as np
from pathlib import Path
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from tqdm import tqdm

def load_documents_and_embeddings(root_data_path : Path, verbose=False) -> InMemoryDocumentStore:
    """
    Load metadata and embeddings from disk and recreate documents 
    returning a InMemoryDocumentStore.
    """
    metadata_path = root_data_path / "haystack/docs"
    embed_path = root_data_path / "haystack/embeddings"
    # Initialize an empty document store
    document_store = InMemoryDocumentStore()
    if verbose:
        print("Reconstructing the InMemoryDocumentStore...")

    for textnmeta_file in tqdm(list(metadata_path.glob("*.gz"))):
        embedings_file = embed_path / f"{textnmeta_file.stem}.npz"
        if not embedings_file.exists():
            print(f"Warning: Embeddings file missing for {textnmeta_file}. Skipping...")
            continue
        with textnmeta_file.open("rb") as f: # Load metadata
            compressed_metadata = f.read()
            metadata = json.loads(gzip.decompress(compressed_metadata).decode("utf-8"))
        embeddings_data = np.load(embedings_file) # Load embeddings
        embeddings = embeddings_data["embeddings"]
        # Ensure the counts match
        if len(metadata) != len(embeddings):
            print(f"Error: Metadata and embeddings count mismatch in {textnmeta_file}. Skipping...")
            continue
        documents = []
        for meta, embedding in zip(metadata, embeddings):
            documents.append(Document(content=meta["content"], meta=meta["meta"], embedding=embedding))
        document_store.write_documents(documents)
    if verbose:
        print(f"Reconstruction complete. {len(document_store.storage)} documents loaded into the store.")
    return document_store

