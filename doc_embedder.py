import json
import numpy as np
from pathlib import Path
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.utils import ComponentDevice
from tqdm import tqdm 
import gzip
from typing import List
import pandas as pd 
import spacy
from copy import deepcopy 
from doc_cleaner import clean_sentences, count_words

# Configuration
ROOT_PATH = Path("/mnt/shared/ipp/") 
DOCS_PATH = Path("/mnt/shared/ipp/text_clean") 
OUTPUT_DOC_DIR = ROOT_PATH / "haystack/docs"
OUTPUT_EMBED_DIR = ROOT_PATH / "haystack/embeddings/"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # faster: "sentence-transformers/all-MiniLM-L6-v2"

# Ensure output directories exist
OUTPUT_DOC_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_EMBED_DIR.mkdir(parents=True, exist_ok=True)

def clean_and_split_text(text, nlp, word_count=200, overlap=30):
    """
    Splits text into coherent chunks suitable for embeddings in a RAG pipeline.

    Args:
        text (str): The input text (noisy sermon content).        
        word_count (int): Desired number of words per chunk.
        overlap (int): Number of words to overlap between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    # Step 2: Sentence tokenization
    sentences = clean_sentences(text.replace('\n', ' '), return_list=True)

    # Step 3: Dynamic merging of sentences to meet word count thresholds
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = count_words(sentence)
        
        # If adding the sentence exceeds the word_count limit, finalize the chunk
        if current_word_count + sentence_word_count > word_count:
            if current_word_count >= word_count:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
        
        # Add the sentence to the current chunk
        current_chunk.append(sentence)
        current_word_count += sentence_word_count

    # Add the last chunk if it's non-empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    adjusted_chunks = []
    for chunk in chunks:
        chunk_word_count = count_words(chunk)
        
        if chunk_word_count > 1.25 * word_count:
            # Use spaCy tokens to split chunk into smaller segments
            doc = nlp(chunk)
            words = [token.text for token in doc if not token.is_punct]  # Exclude punctuation
            
            # Break into roughly equal parts close to word_count
            num_splits = (chunk_word_count + word_count - 1) // word_count  # Ceiling division
            split_size = chunk_word_count // num_splits

            start_idx = 0
            for _ in range(num_splits):
                end_idx = start_idx + split_size
                # Create a new chunk from tokens[start_idx:end_idx]
                adjusted_chunk = ' '.join(words[start_idx:end_idx])
                adjusted_chunks.append(adjusted_chunk)
                start_idx = end_idx
        else:
            adjusted_chunks.append(chunk)
    return adjusted_chunks


class SentenceWordSplitter(DocumentSplitter):
    """
    Splits text into coherent chunks suitable for embeddings in a RAG pipeline.
    Starts with sentences but then splits into chunks of specified length in words with overlap
    """
    def __init__(self, word_count=200, overlap=30, spacy_model="pt_core_news_lg"):
        super().__init__()
        # Load the SpaCy model for sentence segmentation
        self.word_count = word_count
        self.overlap = overlap
        self.nlp = spacy.load(spacy_model) # Use pt_core_news_lg for a more accurate model
        

    def _split(self, document: Document):
        # Initialize the list to store split sentences
        sentences = []        
        chunks = clean_and_split_text(document.content, self.nlp, 
                                      word_count=self.word_count, overlap=self.overlap)
        split_docs = []
        for i in range(0, len(chunks)-1):
            current_words = chunks[i].split()       
            next_words = chunks[i+1].split()[:self.overlap] # first overlap words of next chunk
            chunk = ' '.join(current_words + next_words)
            meta = deepcopy(document.meta)
            meta['split_id'] = i
            meta["split_idx_start"] = i
            meta["source_id"] = document.id 
            split_docs.append(
                Document(content=chunk, meta=meta)
            )            
        return split_docs

    def run(self,  documents: List[Document]):
        split_docs = []
        for doc in documents:
            split_docs.extend(self._split(doc))
        return {"documents": split_docs}

    
document_splitter = SentenceWordSplitter(word_count=200, overlap=30)
document_embedder = SentenceTransformersDocumentEmbedder(
    model=MODEL_NAME, device=ComponentDevice.from_str("cuda:0"), progress_bar=False
)

# metadata to include on vector embeddings
metadatacsv = pd.read_csv(ROOT_PATH / 'metadata.txt')
metadatacsv = metadatacsv[['artist', 'url', 'spot_url', 'sentences_min', 'duration', 'sdcl_file_name']]
metadatacsv.duration = metadatacsv.duration * 0.001 / 60
metadatacsv = metadatacsv.rename(columns={'url' : 'sdc_url', 'sentences_min' : 'sent_min'})
document_embedder.warm_up() # [Document...] now contain embeedings    

# Process each document
for path in tqdm(list(DOCS_PATH.glob("*.txt"))):    
    embedings_file = OUTPUT_EMBED_DIR / f"{path.stem}.npz"
    textnmeta_file = OUTPUT_DOC_DIR / f"{path.stem}.gz"
    # Skip if both files already exist - some names includes '.' so
    if embedings_file.exists() and textnmeta_file.exists():
        continue
    # Read the document content
    with path.open("r") as f:    
        text = f.read()        
    text = text.replace('\n', ' ') # just in this case 
    row = metadatacsv.query(f"'{path.stem}' in sdcl_file_name")
    meta = row.to_dict('records')[0]
    del meta['sdcl_file_name']
    meta['title'] = path.stem
    doc = Document(content=text, meta=meta)
    # need more metadata from spotify, soundcloud etc.
    split_docs = document_splitter.run([doc])  # better split
    # need a good cleaning and maybe another splitter?
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
        compressed_metadata = gzip.compress(json.dumps(metadata, ensure_ascii=False).encode("utf-8"))
        f.write(compressed_metadata)
        
print(f"Processed all documents from {DOCS_PATH}. Results saved to {OUTPUT_DOC_DIR} and {OUTPUT_EMBED_DIR}.")
