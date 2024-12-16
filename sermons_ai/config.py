import pathlib

DATAPATH = pathlib.Path('/mnt/Data/ipp-sermons-texts')

config = {}
config['debug'] = False
config['path'] = {}
config['path']['metadata'] = DATAPATH / 'metadata'
config['path']['transcripts'] = {}
config['path']['transcripts']['alignment'] = DATAPATH / 'transcripts' / 'alignment'
config['path']['transcripts']['raw'] = DATAPATH / 'transcripts' / 'raw'
config['path']['transcripts']['processed'] = DATAPATH / 'transcripts' / 'processed'
config['path']['audio'] = DATAPATH / 'audio'
config['path']['wav'] = config['path']['audio'] / 'wav'
config['metadata_csv'] = config['path']['metadata'] / 'metadata.csv'
config['preacher_names_csv'] = config['path']['metadata'] / 'preacher_names.txt'
config['spotify'] = {}
config['spotify']['client_id'] = 'set-at-run-time'
config['spotify']['client_secret'] = 'set-at-run-time'
config['spotify']['show_id'] = '1DgxzkzYvNGLNv7UawbEUP' # ipperegrinos podcast's unique Spotify ID
config['whisperx'] = {}
config['whisperx']['worker_script'] = pathlib.Path('/home/andre/Projects/ipp-sermons/sermons_ai/transcribex_worker.py')
# to join with url soundcloud and spotify
config['soundcloud_base_url'] = "https://soundcloud.com/ipperegrinos"
config['spotify_base_url'] = "https://open.spotify.com/episode"

config['yt-dlp-cmd'] = "yt-dlp https://soundcloud.com/ipperegrinos -x --audio-format best --write-info-json" # --playlist-end 30

config['doc_cleaner'] = {}
config['doc_cleaner']['language_tool_rules'] = [ "UPPERCASE_AFTER_COMMA",
												"UPPERCASE_SENTENCE_START",
												"VERB_COMMA_CONJUNCTION",
												"ALTERNATIVE_CONJUNCTIONS_COMMA",
												"PORTUGUESE_WORD_REPEAT_RULE" ]

# RAG with haystack
config['rag'] = {}
config['rag']['models'] = {}
config['rag']['models']['sentence-transformer'] = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 
config['rag']['models']['reranker-transformer'] = "BAAI/bge-reranker-v2-m3"
# faster: "sentence-transformers/all-MiniLM-L6-v2"
config['path']['rag'] = {}
config['path']['rag']['embeddings'] = DATAPATH / 'haystack' / 'embeddings'
config['path']['rag']['docs'] = DATAPATH / 'haystack' / 'docs'

