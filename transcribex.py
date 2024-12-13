import subprocess
from pathlib import Path
from tqdm import tqdm
import sys 
import pandas as pd 
import subprocess
from pathlib import Path
from tqdm import tqdm
from config import config

def audio_to_wav(input_file: Path):
    """
    Process the audio file by applying loudness normalization, noise reduction,
    and a high-pass filter, then convert directly to WAV for whisperx
    """    
    wav_file = config['path']['wav'] / f"{input_file.stem}.wav" 
    
    ffmpeg_command = [
        'ffmpeg', '-i', str(input_file),
        '-vn',  # Exclude video
        '-af', 'loudnorm,afftdn=nf=-25,highpass=f=150', 
        '-ar', '16000',  # Whisper-required sample rate
        '-ac', '1',      # Mono channel
        str(wav_file)
    ]
    try:
        subprocess.run(ffmpeg_command, check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_file}: {e}", file=sys.stderr, flush=True)
    return wav_file

def transcribe_audio(wav_file: Path):    
    
    output_path = config['path']['transcripts']['raw'] 
    cmd = [
        "python", str(config['whisperx']['worker_script']),
        "--audio-file", str(wav_file),
        "--output-path", str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in transcription subprocess for {wav_file}: {e}")


if __name__ == "__main__":    
    config['path']['wav'].mkdir(exist_ok=True)
    config['path']['transcripts']['raw'].mkdir(exist_ok=True)
    
    metadata = pd.read_csv(config['metadata_csv'])
    # get only the ones that haven't been transcribed yet 
    selected = metadata[~metadata.transcribed]  #  (transcribed == False)

    for idx, row in tqdm(selected.iterrows()):
        audio = config['path']['audio'] / row['audio']
        wav_file = audio_to_wav(audio)
        transcribe_audio(wav_file)
        metadata.loc[idx, 'wav'] = True
        metadata.loc[idx, 'transcribed'] = True
        metadata.to_csv(config['metadata_csv'], index=False)



