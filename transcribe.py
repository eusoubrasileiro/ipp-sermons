# pip install scdl
# this downloads all tracks from that user to the path specified
# -a download all
# -c skip and continue those already downloaded 
# scdl -l https://soundcloud.com/ipperegrinos -a -c --path /home/andre/music/ipperegrinos
import subprocess
import threading
from pathlib import Path
from tqdm import tqdm
import sys 
import random 

data_folder = Path('/mnt/shared/ipp')
whispercpp_path = Path("/home/andre/whisper.cpp/")

def process_audio(input_file: Path):
    """
    Process the audio file by applying loudness normalization, noise reduction,
    and a high-pass filter using FFmpeg. Then convert to WAV format for Whisper.
    """    
    output_file = data_folder / 'processed' / (input_file.stem + '.mp3')
    if not output_file.exists():
        # Step 1: Apply audio processing with FFmpeg and save as MP3
        ffmpeg_command = [
            'ffmpeg', '-i', str(input_file),    
            '-vn',  # Exclude the video stream (cover art)
            '-af', 'loudnorm, afftdn=nf=-25, highpass=f=150', 
            '-ar', '16000',  # Set audio sample rate to 16kHz - microphone maximum 16kHz
            '-ac', '1',  # Set audio channels to mono
            '-b:a', '64k', # enough bit rate
            str(output_file)
        ]
        try:
            subprocess.run(ffmpeg_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {input_file}: {e}", file=sys.stderr, flush=True)    

    wav_output_file = data_folder / 'wav' / (output_file.stem + '.wav')
    if not wav_output_file.exists():
        # Step 2: Convert the processed MP3 to WAV for Whisper (16kHz, mono)
        ffmpeg_wav_command = [
            'ffmpeg', '-i', str(output_file),
            '-ar', '16000',  # Set sample rate to 16kHz
            '-ac', '1',      # Set audio channels to mono
            str(wav_output_file)
        ]
        try:
            subprocess.run(ffmpeg_wav_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error converting {output_file} to WAV: {e}", file=sys.stderr, flush=True)
    return wav_output_file


def run_whisper_transcription(input_file: Path, use_gpu: bool = False):
    """
    Run whisper.cpp transcription on the specified audio file, optionally with GPU support.
    """
    output_path = data_folder / "text" / (input_file.stem + ".txt")
    if not output_path.exists():
        whisper_command = [
            f"{str(whispercpp_path/'main')}",
            "-t", "6",
            "-l", "pt",
            "-m", f"{str(whispercpp_path/'models/ggml-large-v3-turbo.bin')}",
            "-f", str(input_file),
            "-np", # No printings
            "-ng", # Disable GPU by default
        ]
        if use_gpu:
            whisper_command.remove("-ng")  # Remove - to enable GPU mode    
        try:
            with open(output_path, "w") as f:
                subprocess.run(whisper_command, stdout=f, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error transcribing {input_file}: {e}")


# using only one gpu and fine I don't care about the time it will take
if __name__ == "__main__":
    mp3_folder = data_folder / 'mp3'
    mp3s = list(mp3_folder.glob("*.mp3"))    
    for file in tqdm(mp3s):
        processed_file = process_audio(file)
        run_whisper_transcription(processed_file, use_gpu=True)