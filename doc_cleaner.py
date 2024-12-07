import pathlib
import language_tool_python
from tqdm.notebook import tqdm  # Use tqdm.notebook for Jupyter
import re 
import spacy
import pandas as pd # for reading duration times 

nlp = spacy.load("pt_core_news_lg")
language_tool = None 

def recursive_correct(text):
    ncorrections = 0            
    while True:
        matches = language_tool.check(text)
        ncorrections += len(matches)
        if not matches:
            break
        text = language_tool_python.utils.correct(text, matches)
    return text, ncorrections

def read_transcript(path, save=False):
    with path.open('r') as f:
        text = f.readlines()
    ctext = ''
    for line in text:        
        line = re.sub('\[.+\]\s+', '', line) # remove transcript timestamp mark []
        ctext += line.replace('\n', ' ') 
    return ctext 

def count_sent(text):
    doc = nlp(text)
    return len(list(doc.sents))

def count_words(text):
    wcount = 0
    for sent in nlp(text).sents:
        for token in sent: # words or symbols like punctuation        
            if not token.is_punct:
                wcount += 1
    return wcount


def count_words_sent(sent):
    wcount = 0
    for token in sent: # words or symbols like punctuation        
        if not token.is_punct:
            wcount += 1
    return wcount

def text_long_sentence_split(text : str, verbose=False) -> str:
    """
    Uses spaCy to split overly long sentences based on conjunctions and clauses.
    """        
    cc = [ "ou", "e", "mas", "portanto", 
          "porém", "entretanto", "assim", "pois", "logo"]  # Coordinating conjunction (e.g., "and", "but")
    doc = nlp(text)    
    if verbose:
        print(f"sentences before split: {count_sent(text)}") 
    corrected_sentences = []
    for sent in doc.sents:
        # If sentence is too long, split further on conjunctions
        if count_words_sent(sent) > 30:
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

def clean_sentences(text, return_list=False) -> str:
    """
    Cleans and refactors sentences using spaCy:
    - Removes leading punctuation at the start of sentences.
    - Capitalizes the first word of each sentence.
    """
    spacy_new = []
    doc = nlp(text)

    for sent in doc.sents:
        tokens = list(sent)  # Get tokens for the sentence
        if tokens[0].is_punct:  # Skip sentences that start with punctuation
            sent = sent[1:]  # Skip the first token if it's punctuation

        # Capitalize the first word
        sentence_text = sent.text.strip()
        if sentence_text:
            capitalized_sentence = sentence_text[0].capitalize() + sentence_text[1:]
            spacy_new.append(capitalized_sentence)
    
    if return_list:
        return spacy_new
    return " ".join(spacy_new)


def clean_transcript(path, save=False, verbose=False):
    text  = read_transcript(path)
    if verbose:        
        print(f"{count_sent(text):3d} starting")
    text1 = clean_sentences(text)
    if verbose:
        print(f"{count_sent(text1):3d} after clean_sentences")
    text2, ncorrections = recursive_correct(text1.replace('\n', ' '))
    if verbose:
        print(f"{count_sent(text2):3d} after recursive_correct {ncorrections:3d}")
    text3 = text_long_sentence_split(text2)
    if verbose:
        print(f"{count_sent(text3):3d} after text_long_sentence_split")
    corrected_text = recursive_correct(text3.replace('\n', ' '))[0]
    if verbose:
        print(f"{count_sent(corrected_text):3d} final after recursive_correct")
    return corrected_text
        

if __name__ == "__main__":    
    
    language_tool = language_tool_python.LanguageTool('pt-BR')
    language_tool.enabled_rules_only = True
    language_tool.enabled_rules = [
            "UPPERCASE_AFTER_COMMA",
            "UPPERCASE_SENTENCE_START",
            "VERB_COMMA_CONJUNCTION",
            "ALTERNATIVE_CONJUNCTIONS_COMMA",
            "PORTUGUESE_WORD_REPEAT_RULE"
        ]

    workpath = pathlib.Path('/mnt/shared/ipp-sermons-text')  
    metadata = pd.read_csv(workpath / 'metadata'/'metadata.txt')
    # metadata.loc[:, 'grammar_score'] = None
    # metadata.loc[:, 'short_score'] = None
    # metadata.loc[:, 'repetition_score'] = None
    metadata.loc[:, 'sentences_min'] = None
    metadata.loc[:, 'sentences'] = None

    files = list((workpath / 'text').glob('*.txt'))
    for file in tqdm(files, desc="Processing Files", position=0):

        text = clean_transcript(file)
        output = workpath / 'text_clean' / file.name.replace('.txt', '.txt')
        row = metadata.query(f"'{file.stem}' in sdcl_file_name")
        duration_mins = row.duration.values.astype(float)[0]*0.001/60.  
        # Process and evaluate
        # corrected_text, grammar_score, short_score, repetition_score, total_sentences = process_and_evaluate_transcript(text)
        # metadata.loc[row.index, 'grammar_score'] = grammar_score
        # metadata.loc[row.index, 'short_score'] = short_score
        # metadata.loc[row.index, 'repetition_score'] = repetition_score
        total_sentences = count_sent(text)
        metadata.loc[row.index, 'sentences'] = total_sentences
        metadata.loc[row.index, 'sentences_min'] = total_sentences/duration_mins    
        #print(file.stem)
        corrected_text = clean_transcript(file, verbose=True)        
        with output.open("w") as f:
            f.write('\n'.join(clean_sentences(corrected_text, return_list=True))) # better to QC
        
    metadata.sort_values('date', axis=0).to_csv( (workpath/'metadata.txt').absolute(), index=False)

     

