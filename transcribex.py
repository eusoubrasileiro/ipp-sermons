# pip install scdl
# this downloads all tracks from that user to the path specified
# -a download all
# -c skip and continue those already downloaded 
# scdl -l https://soundcloud.com/ipperegrinos -a -c --no-playlist-folder
# use tmux and -o (offset) parameter to split the download between multiple scdl instances
import subprocess
from pathlib import Path
from typing import List
from tqdm import tqdm
import sys 
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse


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


def transcribe_audio_subprocess(audio_files: List[Path], script_path : Path, output_path: Path, offset=0):
    for file in tqdm(audio_files[offset:]):
        cmd = [
            "python", script_path.absolute(),
            "--audio-file", str(file.absolute()),
            "--output-path", str(output_path.absolute())
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in transcription subprocess for {file}: {e}")


# using only one gpu and fine I don't care about the time it will take
# root_path = Path('/mnt/shared/ipp')
# whisper_path = Path("/mnt/Data/whisper.cpp")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
                                     Preprocess audio mp3s files and transcribe using whisperx
                                     The intended usage is audio preprocessing first and then transcribing.
                                     That's due preprocessing with ffmpeg be very CPU intensive only
                                     and transcribing being GPU intensive only. Hence they can run separated.
                                     """)
    parser.add_argument("-mp3", "--mp3s-path", default=".", help="path from where mp3's will be read")
    parser.add_argument("-wav", "--wavs-path", default=".", help="path from where wav's will be read")
    parser.add_argument("-out", "--output-root-path", help="root path where output 'wav' and 'text' and files will be created")
    parser.add_argument("-o", "--offset", type=int, default=0, help="start offset: number of mp3/wav files to skip to start wav convertion or transcribe")  
    parser.add_argument("-t", "--transcribe", default=False, action="store_true", help="whether to transcribe")  
    parser.add_argument("-a", "--audio-wav", default=False, action="store_true", help="whether to preprocess audio and convert to wav")      

    args = parser.parse_args()    
    root_path = Path(args.output_root_path)           
    def preprocess_audio():
        mp3s = list(Path(args.mp3s_path).glob("*.mp3"))  
        for file in tqdm(mp3s[args.offset::]):
            process_audio(file, root_path)

    if args.audio_wav:
       preprocess_audio()
    else:
        if args.transcribe:
            wavs = list(Path(args.wavs_path).glob("*.wav")) 
            transcribe_audio_subprocess(wavs,
                                        Path(__file__).resolve().parent / "transcribex_worker.py", 
                                        Path(args.output_root_path).absolute(), 
                                        args.offset)


# python3 transcribex.py -wav /mnt/Data/ipp-sermons/wav -out /mnt/Data/ipp-sermons/ -t
