import pathlib

DATAPATH = pathlib.Path('/mnt/Data/ipp-sermons-texts')

config = {}
config['debug'] = False
config['path'] = {}
config['path']['metadata'] = DATAPATH / 'metadata'
config['path']['transcripts'] = {}
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
config['whisperx']['worker_script'] = pathlib.Path('/home/andre/Projects/ipp-sermons/transcribex_worker.py')

# to join with url soundcloud and spotify
config['sc_base_url'] = "https://soundcloud.com/ipperegrinos"
config['sp_base_url'] = "https://open.spotify.com/episode"
config['yt-dlp-cmd'] = "yt-dlp https://soundcloud.com/ipperegrinos -x --audio-format best --write-info-json" # --playlist-end 30

