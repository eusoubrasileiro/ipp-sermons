"""The two decisions that let a second machine join a transcription run.

Same contract as test_postprocess.py: no GPU, no network, no model. Both
functions under test are pure for exactly that reason -- the GPU-dependent part
is `ctranslate2.get_supported_compute_types`, whose answer is passed in.
"""

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
