import copy
import pathlib
import re
import pandas as pd
import language_tool_python
from tqdm import tqdm
import spacy
from config import config
import gzip
import json
import string

def audio_wav2vec_score(file_path):
    """
    Uses the forced alignment (from wav2vec model) probability score saved as json (gzip).
    With that, for each word of entire transcription, calculates the overal weighted average
    score. 
    The closer to 1 the better. In realy values close to 0.8 are already pretty good.
    Trash audios are close or bellow 0.5

    Details:
        For each word in a segment, calculates the weighted score using the word length as the weight.
        Sums up the weighted scores and weights.
        Calculates and the segment's weighted average score.    
        Aggregates the weighted scores and weights for all segments.
        Calculates and the overall weighted average score.
    """
    # Open, decompress, and load the JSON file
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        data = json.load(file)
    
    segments = data.get("segments", [])
    overall_score_sum = 0
    overall_weight_sum = 0

    # Function to remove punctuations
    def clean_word(word):
        return word.translate(str.maketrans('', '', string.punctuation))

    for segment in segments:
        segment_score_sum = 0
        segment_weight_sum = 0

        for word_info in segment.get("words", []):
            word = clean_word(word_info["word"])
            score = word_info.get("score", 0)
            weight = len(word)
            segment_score_sum += score * weight
            segment_weight_sum += weight

        overall_score_sum += segment_score_sum
        overall_weight_sum += segment_weight_sum

    # Calculate overall weighted average score
    if overall_weight_sum > 0:
        overall_weighted_average = overall_score_sum / overall_weight_sum
    else:
        overall_weighted_average = 0

    return round(100*overall_weighted_average, 1)

# --- Helper Functions ---
def setup_nlp_tools():
    """Setup language and NLP tools."""
    nlp = spacy.load("pt_core_news_lg", disable=["ner", "textcat"])
    language_tool = language_tool_python.LanguageTool('pt-BR')
    language_tool.enabled_rules_only = True
    language_tool.enabled_rules = config['doc_cleaner']['language_tool_rules']
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


def process_file(txt_file, nlp, language_tool, verbose=False):
    # Clean transcript
    cleaned_text, words, sent0, sent1, ncorrec = clean_transcript(txt_file, nlp, language_tool)                
    # Save cleaned transcript
    output_text = config['path']['transcripts']['processed'] / txt_file.name
    processed_sentences = clean_sentences(cleaned_text, nlp, return_list=True)
    with output_text.open("w") as f:
        f.write(" ".join(processed_sentences))
    return words, sent1, sent1/sent0


def process_transcripts():
    """try to clean transcript text files and update metadata."""

    # Setup NLP tools
    nlp, language_tool = setup_nlp_tools()

    config['path']['transcripts']['processed'].mkdir(exist_ok=True)
    metadata = pd.read_csv(config['metadata_csv'])
    
    # get only the ones that haven't been transcribed yet 
    selected = metadata[metadata.transcribed]  #  (transcribed == True)
    selected = selected[~selected.processed]  #  (processed == False)    
    for idx, row in tqdm(selected.iterrows(), desc="Processing Transcript Files", total=len(selected)):

        try: 
            txt_file = (config['path']['transcripts']['raw'] / row['txt']).with_suffix(".txt")
            words, sent, ratio = process_file(txt_file, nlp, language_tool)   

            jsongz_file = (config['path']['transcripts']['alignment'] / row['txt']).with_suffix(".gz")        
            score = audio_wav2vec_score(jsongz_file)        

            metadata.loc[idx, 'score'] = score
            metadata.loc[idx, 'words'] = words            
            metadata.loc[idx, 'sent_ratio'] = ratio
            metadata.loc[idx, 'sentences'] = sent
            metadata.loc[idx, 'words_min'] = words / (row['duration']/60)
            metadata.loc[idx, 'sentences_min'] = sent / (row['duration']/60)
            metadata.loc[idx, 'processed'] = True        
            metadata.to_csv(config['metadata_csv'], index=False)
        except:
            print(f"Error processing {row}")
            metadata.loc[idx, 'processed'] = False
            metadata.to_csv(config['metadata_csv'], index=False)



if __name__ == "__main__":
    process_transcripts()

