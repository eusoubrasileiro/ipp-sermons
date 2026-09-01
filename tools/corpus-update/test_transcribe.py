"""The two decisions that let a second machine join a transcription run.

Same contract as test_postprocess.py: no GPU, no network, no model. Both
functions under test are pure for exactly that reason -- the GPU-dependent part
is `ctranslate2.get_supported_compute_types`, whose answer is passed in.
"""

import gzip
import json

import pytest

import config

import transcribe
import whisperx_worker


# --- compute type -----------------------------------------------------------
#
# The existing corpus was transcribed at float16 on a GTX 1660 SUPER (compute
# capability 7.5). A GTX 1060 is 6.1, where CTranslate2 offers no float16 at
# all, and large-v3 in float32 does not fit in 6 GB. So a second box either
# picks a different compute type or cannot help.

TURING = {"float32", "int8", "int8_float16", "int8_float32", "float16"}
PASCAL = {"float32", "int8", "int8_float32"}


def test_prefers_float16_when_the_card_can_do_it():
    """Whatever else changes, a card that can reproduce the corpus settings
    must keep reproducing them -- the z390m's output has to stay bit-for-bit
    the same code path as the 456 transcripts already committed."""
    assert whisperx_worker.pick_compute_type(TURING) == "float16"


def test_falls_back_to_int8_float32_where_float16_does_not_exist():
    """int8_float32 rather than plain int8: the weights are quantised either
    way, but float32 activations keep it closer to the baseline. Measured on
    one 53-minute sermon against the z390m: alignment score 85.9 vs 85.8, and
    0.57% of words different, nearly all of them spellings of the same Hebrew
    proper noun that both boxes are already inconsistent about."""
    assert whisperx_worker.pick_compute_type(PASCAL) == "int8_float32"


def test_an_explicit_request_the_card_supports_is_honoured():
    assert whisperx_worker.pick_compute_type(TURING, "int8") == "int8"


def test_an_impossible_request_raises_rather_than_falling_back():
    """The failure this exists to prevent: CTranslate2 silently substitutes a
    compute type it can do, so asking for float16 on a 6.1 card yields float32,
    an OOM on a 6 GB board, and a run that dies hours in for no stated reason.
    Refusing loudly at the top is the whole point."""
    with pytest.raises(ValueError, match="float16"):
        whisperx_worker.pick_compute_type(PASCAL, "float16")


# --- sharding ---------------------------------------------------------------
#
# Two boxes work the same sorted backlog. They must never pick the same sermon:
# transcription is the expensive stage, and a collision wastes an hour of GPU
# and produces two transcripts of one sermon to reconcile by hand.

WORK = [f"sermon-{i}" for i in range(10)]


def test_shards_are_disjoint_and_cover_everything():
    first = transcribe.shard(WORK, 1, 2)
    second = transcribe.shard(WORK, 2, 2)

    assert set(first) & set(second) == set()
    assert sorted(first + second) == sorted(WORK)


def test_an_uneven_split_still_loses_nothing():
    parts = [transcribe.shard(WORK, i, 3) for i in (1, 2, 3)]

    assert sorted(sum(parts, [])) == sorted(WORK)
    assert all(parts)


def test_the_default_single_shard_is_the_whole_backlog():
    """One box is the normal case and must not have to know about sharding."""
    assert transcribe.shard(WORK, 1, 1) == WORK


def test_shard_one_takes_the_head_so_a_running_box_keeps_its_place():
    """transcribe.py walks the backlog in sorted order, so the box already
    working takes shard 1 and a joining box takes the tail. Any other mapping
    makes the newcomer start where the incumbent is already busy."""
    assert transcribe.shard(WORK, 1, 2)[0] == WORK[0]
    assert transcribe.shard(WORK, 2, 2)[-1] == WORK[-1]


@pytest.mark.parametrize("index,count", [(0, 2), (3, 2), (1, 0), (-1, 3)])
def test_a_nonsensical_shard_is_refused(index, count):
    with pytest.raises(ValueError):
        transcribe.shard(WORK, index, count)


# --- assignment -------------------------------------------------------------
#
# Sharding alone is not enough to keep two boxes apart over time. The backlog
# shrinks from the head as the local box works, so "the first half of what is
# pending" keeps sliding forward and eventually covers sermons the peer was
# handed hours ago. The split therefore has to be written down.


def stage(tmp_path, monkeypatch, *stems: str) -> None:
    """A work directory with `stems` downloaded and nothing transcribed."""
    for name in ("AUDIO_DIR", "RAW_DIR", "ASSIGNED_DIR"):
        (tmp_path / name).mkdir()
        monkeypatch.setattr(config, name, tmp_path / name)
    for stem in stems:
        (tmp_path / "AUDIO_DIR" / f"{stem}.m4a").touch()


def test_pending_is_every_downloaded_sermon_without_a_transcript(tmp_path, monkeypatch):
    stage(tmp_path, monkeypatch, "a [1]", "b [2]")
    (tmp_path / "RAW_DIR" / "a [1].txt").touch()

    assert [p.stem for p in transcribe.pending_audio()] == ["b [2]"]


def test_a_sermon_assigned_to_a_peer_is_not_transcribed_here(tmp_path, monkeypatch):
    """The collision this prevents: the peer is an hour into a sermon that this
    box, having finished the head of its own backlog, now considers next. Both
    spend an hour of GPU on it and the corpus gains two transcripts of one
    sermon for a human to reconcile."""
    stage(tmp_path, monkeypatch, "a [1]", "b [2]", "c [3]")
    (config.ASSIGNED_DIR / "predator.txt").write_text("b [2].m4a\nc [3].m4a\n")

    assert [p.stem for p in transcribe.pending_audio()] == ["a [1]"]


def test_assignments_from_several_peers_all_count(tmp_path, monkeypatch):
    stage(tmp_path, monkeypatch, "a [1]", "b [2]", "c [3]")
    (config.ASSIGNED_DIR / "one.txt").write_text("b [2].m4a\n")
    (config.ASSIGNED_DIR / "two.txt").write_text("c [3].m4a\n")

    assert [p.stem for p in transcribe.pending_audio()] == ["a [1]"]


def test_nothing_is_assigned_when_no_peer_has_ever_run(tmp_path, monkeypatch):
    """The single-machine case has to stay exactly as it was: no peers, no
    assignment directory, the whole backlog."""
    stage(tmp_path, monkeypatch, "a [1]")
    config.ASSIGNED_DIR.rmdir()

    assert [p.stem for p in transcribe.pending_audio()] == ["a [1]"]


# --- provenance -------------------------------------------------------------
#
# With two machines there are two numerical paths, and a transcript does not
# say which one produced it. The run log is the only place that can, and the
# worker's own stdout is captured into a pipe that is read only on failure --
# so the line has to be lifted out of it deliberately.


def test_the_compute_type_is_lifted_out_of_the_captured_worker_output():
    stdout = (
        ">>Performing voice activity detection using Silero...\n"
        "compute_type=int8_float32 on NVIDIA GeForce GTX 1060\n"
        "some other chatter\n"
    )

    assert transcribe.provenance(stdout) == "compute_type=int8_float32 on NVIDIA GeForce GTX 1060"


def test_output_without_a_provenance_line_yields_nothing():
    """Older workers, and any future one that stops printing it, must not crash
    a run that is otherwise fine."""
    assert transcribe.provenance("just the usual whisper chatter\n") is None


# --- coverage ---------------------------------------------------------------
#
# The guard that let nine truncated sermons into production. It was measuring
# the transcript against the WAV it was handed, which is circular: WhisperX
# transcribes whatever audio exists, so a 1-minute download of a 64.7-minute
# class reaches the end of its own file and scores 100%. The denominator has to
# be what SoundCloud says the sermon lasts.


def sermon(tmp_path, monkeypatch, stem: str, *, declared: float | None, last_end: float,
           wav_seconds: float) -> tuple:
    """A finished transcription: an alignment reaching `last_end`, a WAV of
    `wav_seconds`, and optionally a sidecar declaring the sermon's real length."""
    audio, raw = tmp_path / "audio", tmp_path / "raw"
    audio.mkdir(exist_ok=True)
    raw.mkdir(exist_ok=True)
    monkeypatch.setattr(config, "AUDIO_DIR", audio)

    if declared is not None:
        (audio / f"{stem}.info.json").write_text(json.dumps({"duration": declared}))

    gz = raw / f"{stem}.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        json.dump({"segments": [{"end": last_end, "text": "x"}]}, f)

    wav = tmp_path / f"{stem}.wav"
    wav.write_bytes(b"\0" * (44 + int(wav_seconds * 16000 * 2)))
    return gz, wav


def test_a_whole_sermon_covers_its_declared_duration(tmp_path, monkeypatch):
    gz, wav = sermon(tmp_path, monkeypatch, "a [1]", declared=3600, last_end=3580,
                     wav_seconds=3600)
    assert transcribe.coverage(gz, wav) >= transcribe.MIN_COVERAGE


def test_a_truncated_download_no_longer_validates_against_itself(tmp_path, monkeypatch):
    """The production bug, exactly: 1 minute downloaded of a 64.7-minute class.
    Against the WAV this is 100%; against the sermon it is 1.5%."""
    gz, wav = sermon(tmp_path, monkeypatch, "b [2]", declared=64.7 * 60, last_end=60,
                     wav_seconds=60)
    assert transcribe.coverage(gz, wav) < transcribe.MIN_COVERAGE


def test_the_vad_losing_the_back_half_is_still_caught(tmp_path, monkeypatch):
    """The failure the guard was written for: whole audio, partial transcript."""
    gz, wav = sermon(tmp_path, monkeypatch, "c [3]", declared=3600, last_end=60,
                     wav_seconds=3600)
    assert transcribe.coverage(gz, wav) < transcribe.MIN_COVERAGE


def test_it_falls_back_to_the_wav_when_there_is_no_sidecar(tmp_path, monkeypatch):
    """A peer may be handed audio without one; refusing there would stall the
    backlog over a file that is probably fine."""
    gz, wav = sermon(tmp_path, monkeypatch, "d [4]", declared=None, last_end=3580,
                     wav_seconds=3600)
    assert transcribe.coverage(gz, wav) >= transcribe.MIN_COVERAGE


def test_an_alignment_with_no_segments_covers_nothing(tmp_path, monkeypatch):
    gz, wav = sermon(tmp_path, monkeypatch, "e [5]", declared=3600, last_end=0,
                     wav_seconds=3600)
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        json.dump({"segments": []}, f)
    assert transcribe.coverage(gz, wav) == 0.0


# --- transcripts arriving from a peer ---------------------------------------
#
# `coverage` only ever ran inside `transcribe()`. A peer hands back finished
# raw/alignment pairs through `peer.sh collect`, which touched neither -- and
# the peer cannot check for itself, because `dispatch` ships audio without the
# .info.json that says how long the sermon is. 27 truncated downloads came home
# as well-formed transcripts that way.


def collected(tmp_path, monkeypatch, *specs):
    """A work dir holding raw/alignment pairs, as `collect` would leave it."""
    for name in ("AUDIO_DIR", "RAW_DIR", "ALIGNMENT_DIR", "WAV_DIR"):
        (tmp_path / name).mkdir(exist_ok=True)
        monkeypatch.setattr(config, name, tmp_path / name)
    for stem, declared, last_end in specs:
        if declared is not None:
            (tmp_path / "AUDIO_DIR" / f"{stem}.info.json").write_text(
                json.dumps({"duration": declared})
            )
        with gzip.open(tmp_path / "ALIGNMENT_DIR" / f"{stem}.gz", "wt", encoding="utf-8") as f:
            json.dump({"segments": [{"end": last_end, "text": "x"}]}, f)
        (tmp_path / "RAW_DIR" / f"{stem}.txt").write_text("x")


def test_a_peer_transcript_of_truncated_audio_is_discarded(tmp_path, monkeypatch):
    """`a [1]` surviving is the other half: the sweep walks the whole alignment
    directory, so it runs over everything `transcribe()` already passed on this
    box and must be a no-op on all of it."""
    collected(tmp_path, monkeypatch, ("a [1]", 3600, 3580), ("b [2]", 3600, 700))
    assert transcribe.discard_incomplete() == ["b [2]"]
    assert (config.RAW_DIR / "a [1].txt").exists()
    assert (config.ALIGNMENT_DIR / "a [1].gz").exists()
    assert not (config.RAW_DIR / "b [2].txt").exists()
    assert not (config.ALIGNMENT_DIR / "b [2].gz").exists()


def test_both_halves_go_so_the_sermon_is_pending_again(tmp_path, monkeypatch):
    """`pending_audio()` reads RAW_DIR to decide what is left, so a transcript
    kept aside would silently mark the sermon done."""
    collected(tmp_path, monkeypatch, ("b [2]", 3600, 700))
    (config.AUDIO_DIR / "b [2].m4a").touch()
    transcribe.discard_incomplete()
    assert [p.stem for p in transcribe.pending_audio()] == ["b [2]"]


def test_an_alignment_with_no_sidecar_is_left_alone(tmp_path, monkeypatch):
    """Nothing says how long the sermon is, and the .wav coverage would fall
    back to was deleted when transcription finished."""
    collected(tmp_path, monkeypatch, ("c [3]", None, 10))
    assert transcribe.discard_incomplete() == []
    assert (config.ALIGNMENT_DIR / "c [3].gz").exists()


# --- the VAD is the last thing tried ----------------------------------------
#
# Two of nine sermons in one update reached 8% and 5% of their audio at every
# batch size, three attempts each. That is not the intermittent silero misfire
# the retry ladder was built for -- a smaller batch cannot cure a VAD that has
# decided the sermon ends after four minutes, so the ladder could only fail
# more slowly. Both transcribed whole on the VAD the corpus was built with.
#
# pyannote is not the default for the reason whisperx_worker documents: its
# segmentation model does not fit beside large-v3 in float16 on a 6 GB card.
# The fallback pays for the headroom with the compute type the worker already
# measured against this corpus -- 85.9 vs 85.8, 0.57% of words different.


def attempts(monkeypatch, calls: list) -> None:
    """Record the worker command lines a `transcribe()` call would run, and make
    every one of them fail so the whole ladder is walked."""
    import subprocess as sp

    def run(cmd, **kwargs):
        calls.append(cmd)
        return sp.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(transcribe.subprocess, "run", run)


def flag(cmd: list[str], name: str) -> str | None:
    return cmd[cmd.index(name) + 1] if name in cmd else None


def test_every_batch_size_is_tried_on_the_default_vad(tmp_path, monkeypatch):
    calls: list = []
    attempts(monkeypatch, calls)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "ALIGNMENT_DIR", tmp_path)

    transcribe.transcribe(tmp_path / "a [1].wav")

    silero = [c for c in calls if flag(c, "--vad") in (None, "silero")]
    assert [flag(c, "--batch-size") for c in silero] == ["2", "2", "1"]


def test_the_last_attempt_switches_the_vad(tmp_path, monkeypatch):
    """The point of the change: after the ladder, stop varying the batch size
    and vary the thing that actually failed."""
    calls: list = []
    attempts(monkeypatch, calls)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "ALIGNMENT_DIR", tmp_path)

    transcribe.transcribe(tmp_path / "a [1].wav")

    assert flag(calls[-1], "--vad") == "pyannote"
    assert flag(calls[-1], "--compute-type") == "int8_float32"


def test_a_sermon_that_works_first_time_never_reaches_the_fallback(tmp_path, monkeypatch):
    """The fallback costs a whole extra transcription; it must only run when
    every ordinary attempt is exhausted."""
    calls: list = []
    raw, align = tmp_path / "raw", tmp_path / "align"
    raw.mkdir(), align.mkdir()
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "ALIGNMENT_DIR", align)
    monkeypatch.setattr(transcribe, "coverage", lambda gz, wav: 1.0)

    import subprocess as sp

    def run(cmd, **kwargs):
        calls.append(cmd)
        stem = "a [1]"
        (raw / f"{stem}.txt").write_text("text", encoding="utf-8")
        (raw / f"{stem}.gz").write_bytes(b"gz")
        return sp.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(transcribe.subprocess, "run", run)

    assert transcribe.transcribe(tmp_path / "a [1].wav") is True
    assert len(calls) == 1
    assert flag(calls[0], "--vad") in (None, "silero")
