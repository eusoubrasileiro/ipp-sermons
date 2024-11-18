import json
import numpy as np
from pathlib import Path
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.utils import ComponentDevice
from tqdm import tqdm 
import zlib

# Configuration
ROOT_PATH = Path("/mnt/shared/ipp/") 
DOCS_PATH = ROOT_PATH / "text_clean"
OUTPUT_DOC_DIR = ROOT_PATH / "haystack/docs"
OUTPUT_EMBED_DIR = ROOT_PATH / "haystack/embeddings/"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # faster: "sentence-transformers/all-MiniLM-L6-v2"

# Ensure output directories exist
OUTPUT_DOC_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_EMBED_DIR.mkdir(parents=True, exist_ok=True)

document_splitter = DocumentSplitter(split_by="sentence", split_length=4, split_overlap=1)
#document_splitter = DocumentSplitter(split_by="word", split_length=512, split_overlap=32)
document_embedder = SentenceTransformersDocumentEmbedder(
    model=MODEL_NAME, device=ComponentDevice.from_str("cuda:0"), progress_bar=False
)

# Process each document
for path in tqdm(list(DOCS_PATH.glob("*.txt"))):    
    embedings_file = OUTPUT_EMBED_DIR / f"{path.stem}.npz"
    textnmeta_file = OUTPUT_DOC_DIR / f"{path.stem}.meta"
    # Skip if both files already exist - some names includes '.' so
    if embedings_file.exists() and textnmeta_file.exists():
        continue
    # Read the document content
    with path.open("r") as f:    
        text = f.read()        
    doc = Document(content=text, meta={"title": path.name})
    # need more metadata from spotify, soundcloud etc.
    split_docs = document_splitter.run([doc])  # poor split, only by '.' for sentence 
    # need a good cleaning and maybe another splitter?
    document_embedder.warm_up() # [Document...] now contain embeedings    
    docs = document_embedder.run(split_docs['documents']) # a=input list[Document]
    # Save embeddings and metadata
    embeddings = []
    metadata = []
    for doc in docs['documents']:
        metadata.append({            
            "content": doc.content,
            "meta": doc.meta
        })
        embeddings.append(doc.embedding)        
    np.savez_compressed(embedings_file, embeddings=embeddings)    
    with textnmeta_file.open("wb") as f:
        compressed_metadata = zlib.compress(json.dumps(metadata).encode("utf-8"))
        f.write(compressed_metadata)
        
print(f"Processed all documents from {DOCS_PATH}. Results saved to {OUTPUT_DOC_DIR} and {OUTPUT_EMBED_DIR}.")
