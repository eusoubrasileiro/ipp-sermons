"""Transcript cleaning and the metrics derived from it.

A port of `archive/python-gpu/sermons_ai/doc_preproc.py`, kept close to the
original on purpose: `words`, `sentences`, `sent_ratio`, `words_min` and
`sentences_min` are only comparable across the corpus if they are counted the
same way, and `score` is what the indexer filters on.

The one intentional divergence is that LanguageTool is optional. It needs a JVM,
which this box does not ship with; when it is missing the spaCy passes still run
and the text is still segmented and capitalised, so the output degrades in
polish rather than in correctness. `language_tool_available()` reports which
path a run took.
"""

import gzip
import json
import os
import pathlib
import re
import string

import config

_nlp = None
_language_tool = None
_language_tool_tried = False


def nlp():
    global _nlp
    if _nlp is None:
        import spacy

        # ner/textcat are dead weight here: everything downstream only uses the
        # sentence boundaries and the token/punct split.
        _nlp = spacy.load(config.SPACY_MODEL, disable=["ner", "textcat"])
        # Sermon transcripts are one long unbroken string; the 1e6-char default
        # is not always enough and the parser is the only expensive component.
        _nlp.max_length = 4_000_000
    return _nlp


def language_tool():
    """The LanguageTool instance, or None when no JVM can be found."""
    global _language_tool, _language_tool_tried
    if _language_tool_tried:
        return _language_tool
    _language_tool_tried = True

    jre = pathlib.Path(config.JRE_DIR)
    if jre.is_dir():
        os.environ["JAVA_HOME"] = str(jre)
        os.environ["PATH"] = f"{jre / 'bin'}{os.pathsep}{os.environ['PATH']}"

    try:
        import language_tool_python

        tool = language_tool_python.LanguageTool("pt-BR")
        tool.enabled_rules_only = True
        tool.enabled_rules = config.LANGUAGE_TOOL_RULES
        _language_tool = tool
    except Exception as exc:
        print(f"  LanguageTool unavailable ({exc}); running spaCy-only cleaning", flush=True)
        _language_tool = None
    return _language_tool


def language_tool_available() -> bool:
    return language_tool() is not None


def wav2vec_score(gz_path: pathlib.Path) -> float:
    """Mean wav2vec alignment confidence over the whole sermon, 0-100.

    Weighted by word length so that filler tokens ("e", "a") cannot drag a good
    transcription down. Around 0.8 is already a clean recording; below ~0.5 the
    audio is unusable, which is where the corpus cutoff of 50 comes from.
    """
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    score_sum = 0.0
    weight_sum = 0
    for segment in data.get("segments", []):
        for word_info in segment.get("words", []):
            word = word_info["word"].translate(str.maketrans("", "", string.punctuation))
            weight = len(word)
            score_sum += word_info.get("score", 0) * weight
            weight_sum += weight

    return round(100 * (score_sum / weight_sum if weight_sum else 0), 1)


def read_raw(path: pathlib.Path) -> str:
    """Raw WhisperX output is one segment per line; drop any leading `[...]`
    marker and rejoin into the single paragraph the cleaner works on."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return " ".join(re.sub(r"\[.+\]\s+", "", line).strip() for line in lines)


def recursive_correct(text: str, tool) -> tuple[str, int]:
    """Re-check after each pass: fixing one capitalisation can expose the next."""
    if tool is None:
        return text, 0

    import language_tool_python

    ncorrections = 0
    while True:
        matches = tool.check(text)
        if not matches:
            return text, ncorrections
        ncorrections += len(matches)
        text = language_tool_python.utils.correct(text, matches)


def count_sentences(text: str) -> int:
    return len(list(nlp()(text).sents))


def count_words(text: str) -> int:
    return sum(1 for token in nlp()(text) if not token.is_punct)


def clean_sentences(text: str, return_list: bool = False):
    sentences = []
    for sent in nlp()(text).sents:
        s = sent.text.strip()
        if s:
            sentences.append(s[0].capitalize() + s[1:])
    return sentences if return_list else " ".join(sentences)


def split_long_sentences(text: str) -> str:
    """Break run-on sentences at ", e" / "; mas" style joins.

    Whisper punctuates speech sparsely, and a 60-word sentence spans more than
    one idea -- which hurts retrieval, since chunks are built from this text.
    """
    cc = ["ou", "e", "mas", "portanto", "porém", "entretanto", "assim", "pois", "logo"]
    corrected = []

    for sent in nlp()(text).sents:
        if sum(1 for t in sent if not t.is_punct) <= 35:
            corrected.append(sent.text)
            continue

        prev = None
        chunks: list[str] = []
        for token in sent:
            if token.dep_ == "cc" and token.text in cc and prev in [",", ";"] and chunks:
                chunks[-1] = chunks[-1].replace(",", ".").replace(";", ".")
                chunks.append(token.text.capitalize())
            elif token.is_punct:
                if chunks:  # ignore punctuation spaCy left at the sentence start
                    chunks[-1] += token.text
            else:
                chunks.append(token.text)
            prev = token.text
        corrected.append(" ".join(chunks))

    return " ".join(corrected)


def process(raw_path: pathlib.Path, out_path: pathlib.Path) -> tuple[int, int, float]:
    """Clean one raw transcript onto disk. Returns (words, sentences, sent_ratio)."""
    tool = language_tool()

    text = read_raw(raw_path)
    sent_before = count_sentences(text)

    text = clean_sentences(text)
    text, _ = recursive_correct(text, tool)
    text = split_long_sentences(text)
    text, _ = recursive_correct(text, tool)

    words = count_words(text)
    sent_after = count_sentences(text)

    out_path.write_text(" ".join(clean_sentences(text, return_list=True)), encoding="utf-8")
    return words, sent_after, sent_after / sent_before if sent_before else 0.0
