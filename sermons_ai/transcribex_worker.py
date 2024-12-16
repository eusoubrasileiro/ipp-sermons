import whisperx
import gc
import torch
import json
import gzip
from pathlib import Path
import argparse

def transcribe(audio_file : Path, output_path : Path):    

    txt_file = output_path / (audio_file.stem + ".txt")
    json_gz_file = output_path / (audio_file.stem + ".gz")

    device = "cuda"
    batch_size = 4
    try:
        # 1. Transcribe
        model = whisperx.load_model("large-v3", device, compute_type="float16", language="pt")
        audio = whisperx.load_audio(audio_file)
        result_t = model.transcribe(audio, batch_size=batch_size)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # 2. Align transcription
        model_a, metadata = whisperx.load_align_model(language_code=result_t["language"], device=device)
        result_a = whisperx.align(result_t["segments"], model_a, metadata, audio, device=device)
        del model_a
        gc.collect()
        torch.cuda.empty_cache()

        # Save results
        segments = '\n'.join([seg['text'].strip() for seg in result_a['segments']])
        with txt_file.open("w") as f:
            f.write(segments)

        with json_gz_file.open("wb") as f:
            json_data = json.dumps(result_a, indent=4, ensure_ascii=False).encode('utf-8')
            f.write(gzip.compress(json_data))

    except Exception as e:
        print(f"Error transcribing {audio_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single file transcription whisperx script.")
    parser.add_argument("--audio-file", required=True, help="Path to the audio file to transcribe")
    parser.add_argument("--output-path", required=True, help="Root Path to save transcription output")
    args = parser.parse_args()

    transcribe(Path(args.audio_file), Path(args.output_path))
