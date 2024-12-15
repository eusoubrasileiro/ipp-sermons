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
from config import config
from doc_preproc import clean_sentences, count_words



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
    sentences = clean_sentences(text.replace('\n', ' '), nlp, return_list=True)

    # Step 3: Dynamic merging of sentences to meet word count thresholds
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = count_words(sentence, nlp)
        
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
        chunk_word_count = count_words(chunk, nlp)
        
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


if __name__ == "main":

    # Ensure output directories exist
    config['path']['rag']['docs'].mkdir(parents=True, exist_ok=True)
    config['path']['rag']['embeddings'].mkdir(parents=True, exist_ok=True)

    document_splitter = SentenceWordSplitter(word_count=200, overlap=30)
    document_embedder = SentenceTransformersDocumentEmbedder(
        model=config['rag']['models']['sentence-transformer'], 
        device=ComponentDevice.from_str("cuda:0"), progress_bar=False
    )

    # metadata to include on vector embeddings
    metadata = pd.read_csv(config['metadata_csv'])
    selected = metadata[metadata.processed & (metadata.score > 50)]
    columns = ['name', 'description', 'txt', 'artist', 'duration_str', 'id', 'view_count', 
            'sc_suffix_url', 'sp_suffix_url', 
            'date', 'words', 'sentences', 'words_min', 'sentences_min', 'score']
    metadata = selected[columns]

    document_embedder.warm_up() 

    # Process each document
    for idx, row in tqdm(selected.iterrows(), total=len(selected)):    
        doc_file = config['path']['transcripts']['processed'] / row['txt']
        embedings_file = config['path']['rag']['embeddings'] / f"{doc_file.stem}.npz"
        textnmeta_file = config['path']['rag']['docs'] / f"{doc_file.stem}.gz"
        # Skip if both files already exist - some names includes '.' so
        if embedings_file.exists() and textnmeta_file.exists():
            continue
        # Read the document content
        with doc_file.open("r") as f:    
            text = f.read()        
        meta = row.to_dict()
        meta['spotify_url'] = f"{config['spotify_base_url']}/{row['sp_suffix_url']}"
        meta['soundcloud_url'] = f"{config['soundcloud_base_url']}/{row['sc_suffix_url']}"
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