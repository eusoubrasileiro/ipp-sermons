import pathlib
import re 
import pandas as pd 
import datetime 
import requests
from mutagen.mp3 import MP3
from tqdm import tqdm
from rapidfuzz import process, fuzz

# Grab metadata from spotify using Spotify API
def episodes_spotify():
    # Replace these with your Spotify API credentials
    # CLIENT_ID = 
    # CLIENT_SECRET = 
    # SHOW_ID = 

    # Step 1: Get a token
    auth_response = requests.post(
        'https://accounts.spotify.com/api/token',
        {
            'grant_type': 'client_credentials',
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
        }
    )
    access_token = auth_response.json().get('access_token')

    # Step 2: Fetch episode data
    url = f'https://api.spotify.com/v1/shows/{SHOW_ID}/episodes'
    headers = {
        'Authorization': f'Bearer {access_token}',
    }
    params = {
        'limit': 50,  # Max limit per request; paginate if necessary
    }

    episodes = []
    while url:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        episodes.extend(data['items'])
        url = data.get('next')  # For pagination

    return episodes

def get_autor(row):
    def autor(texto):
        if texto is None:
            return None 
        # A regex busca a palavra 'por' seguida de qualquer combinação de palavras (nome do autor).
        match = re.search(r'por\s(.+)', texto)
        if match:
            return match.group(1)  # Retorna o nome do autor
        return None  # Caso não encontre o autor        
    autor_ = autor(row['name'])    
    return autor_ if autor_ else autor(row['description'])

def find_closest_name(name, df_preachers):
    if name:
        return process.extract(name, df_preachers.name.to_list(), scorer=fuzz.ratio, limit=1)[0][0]

def clean_up_mp3_names():
    for path in tqdm(list((workpath).rglob('*.*'))):
        if '�' in path.name:
            path.rename(path.parent / path.name.replace('�', '-'))        

def soundcloud_metadata():
    metadata = pd.DataFrame(columns=['date', 'name', 'mp3_name', 'duration', 'size', 'artist', 'sc_url'])
    # grabing soundclound metadata information from mp3 downloaded
    date_reg = re.compile('(\d{2}).+(\d{2}).+(\d{4})')
    for path in tqdm(list((workpath / 'mp3_raw').glob('*.mp3'))):    
        name = path.name 
        mp3 = MP3(str(path.absolute()))
        duration = mp3.info.length*1000  # Duration in milliseconds    
        metadata.loc[len(metadata)] = [None, 
                                    mp3['TIT2'].text[0], 
                                    path.stem, 
                                    duration, 
                                    path.stat().st_size, 
                                    mp3['TPE1'].text[0],
                                    str(mp3['WOAF'])] # url 
        try: # try extract date if avaliable
            d, m, y = map(int, date_reg.findall(name)[0])        
            #print(f"{d}/{m}/{y} {name[13:].replace('�', '')}", end='\n')
            metadata.loc[len(metadata)-1, 'date'] = datetime.datetime(y, m, d).date()
        except: # there is no date on the name 
            pass 
    return metadata

workpath = pathlib.Path('/mnt/Data/ipp-sermons')

clean_up_mp3_names()
sc_metadata = soundcloud_metadata()

episodes = episodes_spotify()
sp_metadata = pd.DataFrame(columns=['date', 'pub_date', 'name', 'description', 'duration', 'sp_url'])

for episode in episodes:
    # sometimes date is on name, sometimes date is on description 
    full_text = episode['name'] + ' ' + episode['description'] 
    sp_metadata.loc[len(sp_metadata)] = [
            None, 
            episode['release_date'], 
            episode['name'], 
            episode['description'], 
            datetime.timedelta(seconds=int(episode['duration_ms'])).total_seconds(),
            episode['external_urls']['spotify']
        ]
    try: # try extract date if avaliable
        d, m, y = map(int,re.findall(r'(\d{2}).+(\d{2}).+(\d{4})', full_text)[0])                
        sp_metadata.loc[len(sp_metadata)-1, 'date'] = datetime.datetime(y, m, d).date()
    except:
        pass 

#metadata_spotify.duration_ms = metadata_spotify.duration_ms.astype(int)
#metadata_spotify.sort_values(by='date').tail(10)
#metadata_spotify.sort_values('date', axis=0).to_csv( (workpath/'metadata_spotify.txt').absolute(), index=False)

i=0
metadata = sc_metadata.copy()
metadata.loc[:, 'sp_date'] = None
metadata.loc[:, 'sp_url'] = None
metadata.loc[:, 'pub_date'] = None
metadata.loc[:, 'description'] = None
for row in metadata.iterrows():
    name = row[1]['name']
    res = sp_metadata.query(f"'{name}' in name or '{name}' in description")
    if len(res)==1:        
        metadata.loc[row[0], 'sp_date'] = res.iloc[0].date
        metadata.loc[row[0], 'sp_url'] = res.iloc[0].sp_url
        metadata.loc[row[0], 'pub_date'] = res.iloc[0].pub_date        
        metadata.loc[row[0], 'description'] = res.iloc[0].description        
        i+=1
    else:
        if res.name.nunique() == 1:
            print(f'repeated episode {res.name.iloc[0]} ignoring')
            i+=1
        else:
            break 

metadata = metadata[['name', 'description', 'mp3_name', 'artist', 'sc_url',
       'sp_url', 'date', 'pub_date', 'sp_date', 'duration', 'size']]

metadata['artist'] =  metadata.apply(lambda x: get_autor(x), axis=1)

# remove extra spaces on preacher name
metadata.artist = metadata.artist.apply(lambda x: re.sub(r'\s+', ' ', x).strip() if x else x)
# standar name or reference names of preachers
df_preachers = pd.read_csv( (workpath/'preacher_names.txt').absolute())
# match artist names to preacher names using fuzzy matching
metadata.artist = metadata.apply(lambda r: find_closest_name(r.artist, df_preachers), axis=1)

metadata.sort_values('date', axis=0).to_csv( (workpath/'metadata.txt').absolute(), index=False)