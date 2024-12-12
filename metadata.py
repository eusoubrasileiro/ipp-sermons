import re
import pandas as pd
import datetime
import requests
from tqdm import tqdm
from rapidfuzz import process, fuzz
import json 
from config import config
import sys 
import argparse

def spotify_urls():
    """Fetch metadata from Spotify's API.
    and add to the metadata DataFrame."""    
    auth_response = requests.post( # Step 1: Get access token
        'https://accounts.spotify.com/api/token',
        {
            'grant_type': 'client_credentials',
            'client_id': config['spotify']['client_id'],
            'client_secret': config['spotify']['client_secret'],
        }
    )
    auth_response.raise_for_status()
    access_token = auth_response.json().get('access_token')    
    episodes = [] # Step 2: Fetch episodes
    url = f'https://api.spotify.com/v1/shows/{config["spotify"]["show_id"]}/episodes'
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {'limit': 50}  # Adjust for pagination
    while url:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        episodes.extend(data['items'])
        url = data.get('next')  # Next page URL
    sp_meta = [] # Process episode data
    for episode in episodes:
        episode_data = {
            'name_and_description': episode['name'] + ' ' + episode['description'], # for matching
            'id': episode['id'], # the only needed information 
        }
        sp_meta.append(episode_data)
    return pd.DataFrame(sp_meta)  
    
def update_artist(name, preachers, full_names):
    """
    Update the artist column based on preacher names.
    Uses fuzzy matching to find the closest match in the preacher names list.
    """    
    name = str(name).lower() 
    for prefix in ['seminarista', 'presb.', 'pr.', 'rev.', 'pb.', 'presb', 
                'rev', 'rev', 'semi.', 'pres.', 'preb.', 'pastor', 'reverendo']:
        if prefix in name:
            name = name.replace(prefix, '') 
    match_ = process.extractOne(name, preachers, 
                                scorer=fuzz.ratio, score_cutoff=70) 
    return full_names[int(match_[2])] if match_ else None

def try_extract_date(text, timestamp):
    date_reg = re.compile(r'(\d{2}).+(\d{2}).+(\d{4})')
    date = None
    match = date_reg.search(text)
    if match:
        try:
            d, m, y = map(int, match.groups())
            date = datetime.date(y, m, d)
        except: # use date of publication            
            date = pd.to_datetime(timestamp, unit='s').date()       
    return date

def update_soundcloud_metadata(metadata_ids):
    """
    Extract metadata from .json files from yt-dlp folder.
    ONLY audio not on the metadata table (check by id from `metadata_ids`) have metada extraced. 

    - fetch episode metadata from spotify to get url 

    return a dataframe with the new metadata
    """      
    def get_audio(json_meta):
        return [audio for audio in ytdlp_path.glob(f"*{json_meta['display_id']}*") 
                if audio.suffix != '.json'][0]                 
    
    ytdlp_path = config['path']['audio']  
    spotify_meta = spotify_urls()
    # preacher names from csv for standardize the artist column using fuzzy matching
    df_preachers = pd.read_csv(config['preacher_names_csv'])
    preachers = df_preachers.name.str.lower().to_list()
    preachers_full_names = df_preachers.full_name.to_list()

    records=[]
    for file in tqdm(ytdlp_path.glob("*.info.json")):
        json_meta = json.load(file.open('r'))
        if int(json_meta['id']) in metadata_ids: 
            continue
        if json_meta['_type'] == 'video': 
            meta = {
            'name' : json_meta['title'],
            'description' : json_meta['description'],
            'audio_name' : get_audio(json_meta),
            'artist' : json_meta['artist'] if 'artist' in json_meta else None,    
            'sc_suffix_url' : json_meta['webpage_url_basename'],
            'duration_str' : json_meta['duration_string'] if 'duration_string' in json_meta else None,
            'id' : int(json_meta['display_id']),
            'view_count' : json_meta['view_count'],
            'duration' : json_meta['duration'], # is seconds
            'timestamp' : json_meta['timestamp'],
            'date' : try_extract_date(json_meta['title'] + json_meta['description'], 
                                      json_meta['timestamp']),
            }            
            # standardize the preachers names
            meta['artist'] = update_artist(meta['artist'], preachers, preachers_full_names)

            # adding spotify url 
            matching_spotify = spotify_meta[spotify_meta['name_and_description'].str.contains(
                json_meta['title'], na=False, regex=False)]             
            if len(matching_spotify) >= 1:            
                if len(matching_spotify) > 1:
                    print(f"Found multiple matches for '{json_meta['title']}' on spotify : {matching_spotify}",
                          file=sys.stderr)
                # update json_meta with spotify suffix url 
                meta.update({ 'sp_suffix_url' : matching_spotify.iloc[0]['id'] }) 
            else: # for debuggin extreme cases
                print(f"No match for '{json_meta['title']}' to find spotify url", file=sys.stderr)

            meta['transcribed'] = False
            meta['wav'] = False            
            records.append(meta)

    return pd.DataFrame(records)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-spot-id','--spotify-client-id', default=False, action='store_true')
    parser.add_argument('-spot-cs','--spotify-client-secret', default=False, action='store_true')
    args = parser.parse_args()
    #spotify_client_id = 'f0878b5a01664facb6d8553baf30f024'
    #spotify_client_secret = '7150fe04541a4b05b43ff4ebe6bb8ecf'
    config['spotify']['client_id'] = args.spotify_client_id
    config['spotify']['client_secret'] = args.spotify_client_secret
    metadata = pd.read_csv(config['metadata_csv'])
    metadata_ids  = metadata['id'].to_list()
    new_metadata = update_soundcloud_metadata(metadata_ids)
    metadata = pd.concat([metadata, new_metadata], ignore_index=True)
    metadata.to_csv(config['metadata_csv'], index=False)