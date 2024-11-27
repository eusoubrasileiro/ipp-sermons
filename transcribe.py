# pip install scdl
# this downloads all tracks from that user to the path specified
# -a download all
# -c skip and continue those already downloaded 
# scdl -l https://soundcloud.com/ipperegrinos -a -c --no-playlist-folder
# use tmux and -o (offset) parameter to split the download between multiple scdl instances
import subprocess
from pathlib import Path
from tqdm import tqdm
import sys 
import argparse
import threading

def process_audio(input_file: Path, data_folder: Path):
    """
    Process the audio file by applying loudness normalization, noise reduction,
    and a high-pass filter, then convert directly to WAV for whisper.cpp
    """
    wav_folder = data_folder / 'wav'
    if not wav_folder.exists():
        wav_folder.mkdir()
    wav_file = wav_folder / (input_file.stem + '.wav')
    if not wav_file.exists():
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


def run_whisper_transcription(input_file: Path, whispercpp_path: Path, data_folder: Path, 
                              use_gpu: bool = False):
    """
    Run whisper.cpp transcription on the specified audio file, optionally with GPU support.
    """
    text_folder = data_folder / "text"
    txt_file =  text_folder / (input_file.stem + ".txt")
    if not text_folder.exists():
        text_folder.mkdir()
    if not txt_file.exists():
        whisper_command = [
            f"{str(whispercpp_path/'main')}",
            "-t", "4",
            "-l", "pt",
            "-m", f"{str(whispercpp_path/'models/ggml-large-v3-turbo.bin')}",
            "-f", str(input_file),
            "-np", # No printings
            "-ng", # Disable GPU by default
        ]
        if use_gpu:
            whisper_command.remove("-ng")  # Remove - to enable GPU mode    
        try:
            with open(txt_file, "w") as f:
                subprocess.run(whisper_command, stdout=f, check=True,
                              stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error transcribing {input_file}: {e}")


# using only one gpu and fine I don't care about the time it will take
# root_path = Path('/mnt/shared/ipp')
# whisper_path = Path("/mnt/Data/whisper.cpp")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
                                     Preprocess audio mp3s files and transcribe using whisper.cpp
                                     The intended usage is audio preprocessing first and then transcribing.
                                     That's due preprocessing with ffmpeg be very CPU intensive only
                                     and transcribing being GPU intensive only. Hence they can run separated.
                                     """)
    parser.add_argument("-w", "--whispercpp-path", help="whisper.cpp path for main compiled binary")
    parser.add_argument("-mp3", "--mp3s-path", default=".", help="path from where mp3's will be read")
    parser.add_argument("-wav", "--wavs-path", default=".", help="path from where wav's will be read")
    parser.add_argument("-out", "--output-root-path", help="root path where output 'wav' and 'text' and files will be created")
    parser.add_argument("-o", "--offset", type=int, default=0, help="start offset: number of mp3/wav files to skip to start wav convertion or transcribe")  
    parser.add_argument("-t", "--transcribe", default=False, action="store_true", help="whether to transcribe")  
    parser.add_argument("-a", "--audio-wav", default=False, action="store_true", help="whether to preprocess audio and convert to wav")  
    parser.add_argument("-ng", "--no-gpu", default=False, action="store_true", help="force whisper.cpp not use gpu")  

    args = parser.parse_args()
    whisper_path = Path(args.whispercpp_path)
    root_path = Path(args.output_root_path) 
    mp3s = list(Path(args.mp3s_path).glob("*.mp3"))        
    wavs = list(Path(args.wavs_path).glob("*.wav"))  
    def preprocess_audio():
        for file in tqdm(mp3s[args.offset::]):
            process_audio(file, root_path)
    def transcribe_audio(use_gpu=True):                                
        for file in tqdm(wavs[args.offset::]):
            run_whisper_transcription(file, whisper_path, root_path, use_gpu)
    if args.audio_wav:
        threading.Thread(target=preprocess_audio).start()
    else:
        if args.transcribe:
            threading.Thread(target=transcribe_audio, args=(not args.no_gpu,)).start()


