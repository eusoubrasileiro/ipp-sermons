import pathlib
import re
import pandas as pd
import datetime
import requests
from mutagen.mp3 import MP3
from tqdm import tqdm
from rapidfuzz import process, fuzz


# --- Configuration ---
WORKPATH = pathlib.Path('/mnt/Data/ipp-sermons-texts')
METADATA_FILE = WORKPATH / 'metadata' / 'metadata.csv'
PREACHER_NAMES_FILE = WORKPATH / 'metadata' / 'preacher_names.txt'

# --- Utility Functions ---
def load_metadata():
    """Load the metadata CSV, or create a new DataFrame if it doesn't exist."""
    if METADATA_FILE.exists():
        return pd.read_csv(METADATA_FILE)
    else:
        columns = ['name', 'description', 'mp3_name', 'artist', 'sc_url', 'sp_url', 'date', 
                   'pub_date', 'duration', 'size', 'edited']
        return pd.DataFrame(columns=columns)

# --- SoundCloud Metadata ---
def clean_up_mp3_names():
    """Clean up invalid characters in MP3 file names."""
    for path in tqdm(list(WORKPATH.rglob('*.*')), desc="Cleaning MP3 names"):
        if '�' in path.name:
            path.rename(path.parent / path.name.replace('�', '-'))


def extract_soundcloud_metadata():
    """Extract metadata from MP3 files in the SoundCloud folder."""
    metadata = []
    date_reg = re.compile(r'(\d{2}).+(\d{2}).+(\d{4})')

    for path in tqdm((WORKPATH / 'mp3_raw').glob('*.mp3'), desc="Extracting SoundCloud metadata"):
        mp3 = MP3(str(path))
        mp3_data = {
            'name': mp3['TIT2'].text[0],
            'mp3_name': path.stem,
            'artist': mp3['TPE1'].text[0],
            'sc_url': str(mp3['WOAF']),
            'duration': mp3.info.length,
            'size': path.stat().st_size
        }
        try:
            # Extract date from filename if possible
            match = date_reg.search(path.name)
            if match:
                d, m, y = map(int, match.groups())
                mp3_data['date'] = datetime.date(y, m, d)            
        except Exception as e:
            print(f"Error processing {path.name}: {e}")
        metadata.append(mp3_data)
    return pd.DataFrame(metadata)


# --- Spotify Metadata ---
def fetch_spotify_metadata():
    """Fetch metadata from Spotify's API."""
    CLIENT_ID = 'xxxxxxxxxxx'
    CLIENT_SECRET = 'xxxxxxxxxxx'
    SHOW_ID = 'xxxxxxxxxxx'  # The podcast's unique Spotify ID

    # Step 1: Get access token
    auth_response = requests.post(
        'https://accounts.spotify.com/api/token',
        {
            'grant_type': 'client_credentials',
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
        }
    )
    auth_response.raise_for_status()
    access_token = auth_response.json().get('access_token')

    # Step 2: Fetch episodes
    episodes = []
    url = f'https://api.spotify.com/v1/shows/{SHOW_ID}/episodes'
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {'limit': 50}  # Adjust for pagination

    while url:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        episodes.extend(data['items'])
        url = data.get('next')  # Next page URL

    # Process episode data
    metadata = []
    for episode in episodes:
        episode_data = {
            'name': episode['name'],
            'description': episode['description'],
            'sp_url': episode['external_urls']['spotify'],
            'pub_date': episode['release_date'],
            'duration': episode['duration_ms'] / 1000
        }
        metadata.append(episode_data)

    return pd.DataFrame(metadata)


def get_autor(row, df_preachers=None):
    def autor(texto):
        if texto is None:
            return None 
        # A regex busca a palavra 'por' seguida de qualquer combinação de palavras (nome do autor).
        match = re.search(r'(?:por|Por)\s(.+)', texto)
        if match:
            return match.group(1)  # Retorna o nome do autor
        return None  # Caso não encontre o autor        
    autor_ = autor(row['name'])     
    autor_ = autor_ if autor_ else autor(row['description'])
    if not autor_:
        return None
    return process.extract(autor_, df_preachers.name.to_list(), scorer=fuzz.ratio, limit=1)[0][0]


def update_metadata_row(metadata, combined_row, key_column='name'):
    """
    Update or add a row in the metadata DataFrame based on a key column.
    Args:
        metadata (pd.DataFrame): The metadata DataFrame to update.
        combined_row (dict): The row data to merge into the DataFrame.
        key_column (str): The column used to identify rows (default is 'name').
    Returns:
        pd.DataFrame: Updated metadata DataFrame.
    """
    def update_row_with_dict(dataframe, index, dictionary):  
        for key in dictionary.keys():  
            dataframe.loc[index, key] = dictionary.get(key)     
    combined_row['edited'] = False
    nbefore = len(metadata)
    existing_row = metadata[metadata[key_column] == combined_row[key_column]]
    if not existing_row.empty: # Update an existing row
        index = existing_row.index[0]
        existing_row = existing_row.iloc[0].to_dict()
        if existing_row.get('edited', False):  # If manually edited
            # Only add keys not already present in the existing row
            combined_row = {k: v for k, v in combined_row.items() if k not in existing_row}
            if combined_row:
                combined_row['edited'] = True                         
                update_row_with_dict(metadata, index, combined_row) 
        else:
            update_row_with_dict(metadata, index, combined_row) 
    else: # Add as a new row        
        metadata = pd.concat([metadata, pd.DataFrame([combined_row])], ignore_index=True)
    return len(metadata)-nbefore, metadata


# --- Main Processing ---
def process_metadata():
    """Process SoundCloud and Spotify metadata, and save results."""
    clean_up_mp3_names()

    # Load metadata
    metadata = load_metadata()
    first_run = False
    if len(metadata) == 0:
        first_run = True
    # Extract SoundCloud metadata
    sc_metadata = extract_soundcloud_metadata()
    # Fetch Spotify metadata
    sp_metadata = fetch_spotify_metadata()
    df_preachers = pd.read_csv(PREACHER_NAMES_FILE)
    # Merge SoundCloud and Spotify metadata
    imatch = 0
    for _, sc_row in sc_metadata.iterrows():
        #matching_sc = sc_metadata.query(f"'{sp_row['name'].strip()}' in name or '{sp_row['description'].strip()}' in name")
        matching_sp = sp_metadata[sp_metadata['name'].str.contains(sc_row['name'], na=False, regex=False) | 
                                sp_metadata['description'].str.contains(sc_row['name'], na=False, regex=False)] 
        if len(matching_sp) >= 1:            
            if len(matching_sp) > 1:
                print(f"Repeated file uploaded - Found multiple matches for '{sc_row['name']}' : {matching_sp}")
            #matching_sp = matching_sp.iloc[0] if len(matching_sp) > 1 else matching_sp
            sp_row = matching_sp.iloc[0]
            combined_row = sc_row.to_dict() | sp_row.to_dict() # merge 
            combined_row['artist'] = get_autor(combined_row, df_preachers)
            iadded, metadata = update_metadata_row(metadata, combined_row)            
            imatch += iadded
            if iadded == 0 and first_run: 
                # when running for the first time and the metadata is empty
                # those are repeated mp3 files mistakenly uploaded
                print(f"Repeated file uploaded - This was already inserted {combined_row['name']}")        
        else:
            print(f"No match for '{sp_row['name']}'")

    # Save final metadata
    metadata.to_csv(METADATA_FILE, index=False)    

if __name__ == "__main__":
    metadata = process_metadata()
