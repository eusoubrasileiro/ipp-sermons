import json
import numpy as np
from pathlib import Path
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.utils import ComponentDevice
from tqdm import tqdm 
import zlib
from typing import List
import pandas as pd 

# Configuration
ROOT_PATH = Path("/mnt/Data/ipp-sermons/") 
DOCS_PATH = ROOT_PATH / "text_clean"
OUTPUT_DOC_DIR = ROOT_PATH / "haystack/docs"
OUTPUT_EMBED_DIR = ROOT_PATH / "haystack/embeddings/"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # faster: "sentence-transformers/all-MiniLM-L6-v2"

# Ensure output directories exist
OUTPUT_DOC_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_EMBED_DIR.mkdir(parents=True, exist_ok=True)

import spacy
from copy import deepcopy 

class SpaCySentenceSplitter(DocumentSplitter):
    def __init__(self, split_length=4, split_overlap=1, spacy_model="pt_core_news_md"):
        super().__init__(split_length=split_length, split_overlap=split_overlap)
        # Load the SpaCy model for sentence segmentation
        self.nlp = spacy.load(spacy_model) # Use pt_core_news_lg for a more accurate model

    def _split(self, document: Document):
        # Initialize the list to store split sentences
        sentences = []        
        # Process the document with SpaCy
        doc = self.nlp(document.content)
        # Collect the sentences from the SpaCy document
        sentences = [ sent.text.strip() for sent in doc.sents ] 
        # Now split sentences into chunks of specified length with overlap
        split_docs = []
        for i in range(0, len(sentences), self.split_length - self.split_overlap):
            chunk = sentences[i:i + self.split_length]            
            meta = deepcopy(document.meta)
            meta['split_id'] = i
            meta["split_idx_start"] = i
            meta["source_id"] = document.id 
            split_docs.append(
                Document(content=" ".join(chunk), meta=meta)
            )
        return split_docs

    def run(self,  documents: List[Document]):
        split_docs = []
        for doc in documents:
            split_docs.extend(self._split(doc))
        return {"documents": split_docs}

    
document_splitter = SpaCySentenceSplitter(split_length=4, split_overlap=1)
document_embedder = SentenceTransformersDocumentEmbedder(
    model=MODEL_NAME, device=ComponentDevice.from_str("cuda:0"), progress_bar=False
)

metadatacsv = pd.read_csv(ROOT_PATH / 'metadata.txt')
metadatacsv = metadatacsv[['artist', 'url', 'spot_url', 'sentences_min', 'duration', 'sdcl_file_name']]
metadatacsv.duration = metadatacsv.duration * 0.001 / 60
metadatacsv.rename(columns={'url' : 'scdl_url', 'sentences_min' : 'sent_min'})

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
    row = metadatacsv.query(f"'{path.stem}' in sdcl_file_name")
    meta = row.to_dict('records')[0]
    del meta['sdcl_file_name']
    meta['title'] = path.stem
    doc = Document(content=text, meta=meta)
    # need more metadata from spotify, soundcloud etc.
    split_docs = document_splitter.run([doc])  # better split
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
