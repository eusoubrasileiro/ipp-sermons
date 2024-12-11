import copy
import pathlib
import re
import pandas as pd
import language_tool_python
from tqdm import tqdm
import spacy

# --- Helper Functions ---
def setup_nlp_tools(config):
    """Setup language and NLP tools."""
    nlp = spacy.load("pt_core_news_lg", disable=["ner", "textcat"])
    language_tool = language_tool_python.LanguageTool('pt-BR')
    language_tool.enabled_rules_only = True
    language_tool.enabled_rules = config['language_tool_rules']
    return nlp, language_tool


def recursive_correct(text, language_tool):
    """Recursively correct grammar issues in the text."""
    ncorrections = 0
    while True:
        matches = language_tool.check(text)
        ncorrections += len(matches)
        if not matches:
            break
        text = language_tool_python.utils.correct(text, matches)
    return text, ncorrections


def read_transcript(path):
    """Read and clean the transcript text file."""
    with path.open('r') as f:
        lines = f.readlines()
    text = ' '.join([re.sub(r'\[.+\]\s+', '', line).strip() for line in lines])
    return text


def text_long_sentence_split(text : str, nlp, verbose=False) -> str:
    """
    Uses spaCy to split overly long sentences based on conjunctions and clauses.
    """        
    def count_words_sent(sent):
        wcount = 0
        for token in sent: # words or symbols like punctuation        
            if not token.is_punct:
                wcount += 1
        return wcount
    cc = [ "ou", "e", "mas", "portanto", 
          "porém", "entretanto", "assim", "pois", "logo"]  # Coordinating conjunction (e.g., "and", "but")
    doc = nlp(text)    
    if verbose:
        print(f"sentences before split: {count_sent(text)}") 
    corrected_sentences = []
    for sent in doc.sents:
        # If sentence is too long, try to split further on conjunctions
        # if previous token is a ',' or ';' and current token is a 'cc'
        if count_words_sent(sent) > 35:
            prev_token = None
            chunks = []
            for token in sent:                
                if (token.dep_ == "cc" and token.text in cc and 
                    prev_token is not None and prev_token in [',', ';']
                    and chunks):   
                    chunks[-1] = chunks[-1].replace(',', '.').replace(';', '.') # remove , or ;           
                    # Insert a period to start a new sentence
                    chunks.append(token.text.capitalize())
                else:
                    if token.is_punct:
                        if chunks:                            
                            chunks[-1] += token.text
                        # else ignore punctuation in begin of sentence
                        # splitted by spacy
                    else:
                        chunks.append(token.text)
                prev_token = token.text
            corrected_sentences.append(" ".join(chunks))
        else:
            corrected_sentences.append(sent.text)    
    new_text = " ".join(corrected_sentences)
    if verbose:
        print(f"sentences after split: {count_sent(new_text)}")
    return new_text


def clean_sentences(text, nlp, return_list=False):
    """Clean and refactor sentences using spaCy."""
    sentences = []
    for sent in nlp(text).sents:
        cleaned_sentence = sent.text.strip()
        if cleaned_sentence:
            sentences.append(cleaned_sentence[0].capitalize() + cleaned_sentence[1:])
    return sentences if return_list else " ".join(sentences)

def count_sent(text, nlp):
    doc = nlp(text)
    return len(list(doc.sents))

def count_words(text, nlp):
    wcount = 0
    for sent in nlp(text).sents:
        for token in sent: # words or symbols like punctuation        
            if not token.is_punct:
                wcount += 1
    return wcount

def clean_transcript(path, nlp, language_tool, verbose=False):
    """Clean and process the transcript text."""
    text = read_transcript(path)
    sent0 = count_sent(text, nlp)
    text = clean_sentences(text, nlp)
    text, ncorrec0 = recursive_correct(text, language_tool)
    text = text_long_sentence_split(text, nlp, verbose)
    text, ncorrec1 = recursive_correct(text, language_tool)
    nwords = count_words(text, nlp)
    sent1 = count_sent(text, nlp)
    return text, nwords, sent0, sent1, ncorrec0+ncorrec1

from doc_metadata import update_metadata_row
from tools import pretty_duration


def process_file(txt_file, metadata, nlp, language_tool, config, verbose=False):
        # Clean transcript
        cleaned_text, words, sent0, sent1, ncorrec = clean_transcript(txt_file, nlp, language_tool)
                
        row = metadata[metadata.mp3_name == txt_file.stem]
        row = row.iloc[0].to_dict()
        duration_min = row['duration'] / 60

        combined_row = copy.deepcopy(row) # Update metadata        
        combined_row.update({
            "words": words,
            "sentences": sent1,
            "sentences_min": sent1 / duration_min,
            "duration_str" : pretty_duration(row['duration']),
        })

        update_metadata_row(metadata, combined_row)

        # Save cleaned transcript
        output_path = config['processed_folder'] / txt_file.name
        processed_sentences = clean_sentences(cleaned_text, nlp, return_list=True)
        with output_path.open("w") as f:
            f.write(" ".join(processed_sentences))

# --- Main Script ---

from joblib import Parallel, delayed

def process_transcripts(config):
    """Process transcript files and update metadata."""
    # Setup NLP tools
    nlp, language_tool = setup_nlp_tools(config)

    # Load metadata
    metadata = pd.read_csv(config['metadata_path'])

    # Filter files based on metadata entries
    mp3_files = metadata['mp3_name'].dropna().unique()
    txt_files = [config['raw_folder'] / f"{file}.txt" 
                 for file in mp3_files 
                    if (config['raw_folder'] / f"{file}.txt").exists()]

    # Process files
    for txt_file in tqdm(txt_files, desc="Processing Transcript Files"):
        process_file(txt_file, metadata, nlp, language_tool, config)        

    # Save updated metadata
    metadata.sort_values('date', axis=0).to_csv(config['metadata_path'], index=False)



if __name__ == "__main__":
    # Configuration
    config = {
        "metadata_path": pathlib.Path('/mnt/Data/ipp-sermons-texts/metadata/metadata.csv'),
        "raw_folder": pathlib.Path('/mnt/Data/ipp-sermons-texts/raw'),
        "processed_folder": pathlib.Path('/mnt/Data/ipp-sermons-texts/processed'),
        "language_tool_rules": [
            "UPPERCASE_AFTER_COMMA",
            "UPPERCASE_SENTENCE_START",
            "VERB_COMMA_CONJUNCTION",
            "ALTERNATIVE_CONJUNCTIONS_COMMA",
            "PORTUGUESE_WORD_REPEAT_RULE",
        ],
    }

    # Process transcripts
    process_transcripts(config)
