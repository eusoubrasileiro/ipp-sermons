"""WhisperX worker: one wav in, transcript + word alignment out.

Runs under the ml-tools venv's interpreter, not the pipeline's -- that is where
WhisperX, torch/cu118 and the NVIDIA libraries live. It is invoked as a
subprocess by transcribe.py, one sermon per process, because 6 GB of VRAM cannot
hold the ASR model and the wav2vec aligner at once; each has to be loaded and
freed in turn anyway, so there is nothing to keep resident between sermons.

This is `archive/python-gpu/sermons_ai/transcribex_worker.py`, restored, with
one forced divergence: `--vad` defaults to silero where the original used
pyannote, because pyannote's segmentation model does not fit on this 6 GB card
beside large-v3 in float16 and OOMs before transcribing a word. The cost is that
silero has been observed deciding everything after the first few minutes of a
sermon is not speech; alignment then succeeds on the handful of segments it did
find, so the result is a well-formed transcript of a fifth of the sermon.
`transcribe.py` catches that by coverage, and when no batch size cures it asks
for `--vad pyannote` with a compute type that leaves room for it. The full
reasoning sits beside the `load_model` call -- change one, change both.

Failures exit non-zero and write nothing -- a partial transcript that looks
healthy is worse than no transcript.
"""

import argparse
import gc
import gzip
import json
import sys
from pathlib import Path

# What the corpus was made with, and what any card that can manage it keeps
# using. Everything below is about the cards that cannot.
BASELINE_COMPUTE_TYPE = "float16"

# Second choice, for compute capability < 7.0. The weights are quantised either
# way; float32 activations keep the arithmetic closer to the baseline than
# plain int8 does. Measured on a GTX 1060 against the 1660 SUPER, one 53-minute
# sermon: alignment score 85.9 vs 85.8, identical coverage, 0.57% of words
# different -- and nearly all of those are spellings of one Hebrew proper noun
# that both boxes already spell inconsistently within a single transcript.
FALLBACK_COMPUTE_TYPE = "int8_float32"


def pick_compute_type(supported: set[str], requested: str | None = None) -> str:
    """Choose the compute type, or say why the requested one is impossible.

    CTranslate2 does not offer float16 below compute capability 7.0, and it
    does not refuse either -- it quietly substitutes float32, which for
    large-v3 means 6.2 GB of weights on a 6 GB board and an out-of-memory
    failure hours into a run, with nothing in the log connecting the two. So
    the substitution is made here, deliberately and visibly, and an explicit
    request that the device cannot honour is an error rather than a surprise.
    """
    if requested:
        if requested not in supported:
            raise ValueError(
                f"this GPU cannot do {requested}; it supports {sorted(supported)}"
            )
        return requested
    if BASELINE_COMPUTE_TYPE in supported:
        return BASELINE_COMPUTE_TYPE
    if FALLBACK_COMPUTE_TYPE in supported:
        return FALLBACK_COMPUTE_TYPE
    raise ValueError(f"no usable compute type; this GPU offers {sorted(supported)}")


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


DEFAULT_VAD = "silero"


def transcribe(
    wav: Path,
    out_dir: Path,
    batch_size: int,
    requested: str | None = None,
    vad: str = DEFAULT_VAD,
) -> None:
    import ctranslate2
    import torch

    allow_pyannote_checkpoint(torch)

    import whisperx

    device = "cuda"
    compute_type = pick_compute_type(ctranslate2.get_supported_compute_types(device), requested)
    # Printed rather than assumed: this is the one line that tells a reader of
    # the log which machine produced a given transcript and how.
    print(f"compute_type={compute_type} vad={vad} on {torch.cuda.get_device_name(0)}", flush=True)
    audio = whisperx.load_audio(str(wav))

    # silero by default rather than the pyannote VAD the original used, for one
    # reason: pyannote's segmentation model does not fit on this 6 GB card next
    # to large-v3 in float16, and OOMs before it transcribes a word. silero is
    # cheap enough to coexist, at the cost of sometimes deciding that the back
    # half of a sermon is not speech.
    #
    # transcribe.py detects that by length. Where it is intermittent a retry
    # clears it; where it is systematic for one recording no batch size ever
    # will, so the last attempt passes --vad pyannote here and drops the compute
    # type to make room for it.
    model = whisperx.load_model(
        "large-v3", device, compute_type=compute_type, language="pt", vad_method=vad
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
    parser.add_argument(
        "--compute-type",
        default=None,
        help="force a compute type; omit to use float16 where the GPU has it "
        "and int8_float32 where it does not",
    )
    parser.add_argument(
        "--vad",
        default=DEFAULT_VAD,
        choices=["silero", "pyannote"],
        help="voice activity detector; pyannote needs a compute type that "
        "leaves room for its segmentation model",
    )
    args = parser.parse_args()

    try:
        transcribe(
            Path(args.wav), Path(args.out_dir), args.batch_size, args.compute_type, args.vad
        )
    except Exception as exc:
        print(f"Error transcribing {args.wav}: {exc}", file=sys.stderr)
        sys.exit(1)
