"""WhisperX worker: one wav in, transcript + word alignment out.

Runs under the ml-tools venv's interpreter, not the pipeline's -- that is where
WhisperX, torch/cu118 and the NVIDIA libraries live. It is invoked as a
subprocess by transcribe.py, one sermon per process, because 6 GB of VRAM cannot
hold the ASR model and the wav2vec aligner at once; each has to be loaded and
freed in turn anyway, so there is nothing to keep resident between sermons.

This is `archive/python-gpu/sermons_ai/transcribex_worker.py`, restored. The
box's general-purpose `transcreve.py` would have done the same job, but it pins
`vad_method="silero"`, which the original did not: silero has been observed
here deciding that everything after the first five minutes of a sermon is not
speech, and because alignment then succeeds on the handful of segments it did
find, the result is a well-formed transcript of a fifth of the sermon. The
default VAD is what produced the existing 455, so it is what this uses.

Failures exit non-zero and write nothing -- a partial transcript that looks
healthy is worse than no transcript.
"""

import argparse
import gc
import gzip
import json
import sys
from pathlib import Path


def allow_pyannote_checkpoint(torch) -> None:
    """Let the VAD checkpoint load under torch >= 2.6.

    torch 2.6 flipped `torch.load` to `weights_only=True`, which rejects the
    pyannote segmentation checkpoint because it pickles omegaconf config
    objects. WhisperX has no way to pass the flag through, so the default VAD
    simply dies -- which is why the box's general-purpose transcriber switched
    to silero. Restoring the old default for this process is the smaller evil:
    the alternative VAD silently truncates sermons, and the checkpoint here is
    the one WhisperX itself downloads, not user input.
    """
    original = torch.load

    def load(*args, **kwargs):
        # Overridden, not defaulted: Lightning's checkpoint loader passes
        # weights_only=True explicitly, so a default would never be consulted.
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = load


def transcribe(wav: Path, out_dir: Path, batch_size: int) -> None:
    import torch

    allow_pyannote_checkpoint(torch)

    import whisperx

    device = "cuda"
    audio = whisperx.load_audio(str(wav))

    # silero rather than the pyannote VAD the original used, for one reason:
    # pyannote's segmentation model does not fit on this 6 GB card next to
    # large-v3 in float16, and OOMs before it transcribes a word. silero is
    # cheap enough to coexist, at the cost of occasionally deciding that the
    # back half of a sermon is not speech -- which transcribe.py detects by
    # length and retries, because the misfire is intermittent, not systematic.
    model = whisperx.load_model(
        "large-v3", device, compute_type="float16", language="pt", vad_method="silero"
    )
    result = model.transcribe(audio, batch_size=batch_size)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Word-level timestamps and, more importantly, per-word confidence: the
    # `score` column of metadata.csv is a weighted mean over exactly this.
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device=device)
    del align_model
    gc.collect()
    torch.cuda.empty_cache()

    segments = aligned["segments"]
    if not segments:
        raise RuntimeError("alignment produced no segments")

    (out_dir / f"{wav.stem}.txt").write_text(
        "\n".join(seg["text"].strip() for seg in segments), encoding="utf-8"
    )
    with (out_dir / f"{wav.stem}.gz").open("wb") as f:
        f.write(gzip.compress(json.dumps(aligned, indent=4, ensure_ascii=False).encode("utf-8")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    try:
        transcribe(Path(args.wav), Path(args.out_dir), args.batch_size)
    except Exception as exc:
        print(f"Error transcribing {args.wav}: {exc}", file=sys.stderr)
        sys.exit(1)
