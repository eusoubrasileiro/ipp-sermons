"""Step 1 -- list what the church has published that the corpus does not have.

SoundCloud is the source of truth: every row in metadata.csv carries a
SoundCloud track id, so the diff is an id set difference and needs no
credentials. Spotify only ever contributed the `sp_suffix_url` column and is
resolved later, optionally (see spotify_ids.py).
"""

import csv
import json

import yt_dlp

import config


def soundcloud_tracks() -> list[dict]:
    """Every track on the channel, cheaply: extract_flat skips the per-track API
    call, so this is one request instead of ~620."""
    with yt_dlp.YoutubeDL({"extract_flat": True, "skip_download": True, "quiet": True}) as ydl:
        info = ydl.extract_info(config.SOUNDCLOUD_TRACKS_URL, download=False)
    return info["entries"]


def known_ids() -> set[str]:
    with config.METADATA_CSV.open(encoding="utf-8") as f:
        return {row["id"].strip() for row in csv.DictReader(f)}


def main() -> None:
    config.ensure_dirs()

    tracks = soundcloud_tracks()
    have = known_ids()
    pending = [
        {"id": str(t["id"]), "title": t.get("title"), "url": t["url"]}
        for t in tracks
        if str(t["id"]) not in have
    ]
    # Oldest first, so an interrupted backlog run leaves a contiguous gap at the
    # recent end rather than holes scattered through the archive.
    pending.reverse()

    config.PENDING_JSON.write_text(
        json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"soundcloud: {len(tracks)} tracks, corpus has {len(have)}, pending {len(pending)}")
    print(f"wrote {config.PENDING_JSON}")


if __name__ == "__main__":
    main()
