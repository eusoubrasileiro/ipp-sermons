"""SoundCloud track id -> Spotify episode id, for the `sp_suffix_url` column.

Deliberately best-effort. The column is a convenience link on the site, not
something search depends on, and Spotify's API needs a client credential that
this pipeline should not require to run. Two sources, in order:

  1. `sermons/tracks.json`, a scrape someone already did, matched to SoundCloud.
     Free, offline, and covers everything published up to when it was taken.
  2. The Spotify API, only if SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET are in
     the environment. See .env.example.

With neither, rows get an empty sp_suffix_url and the site simply shows no
Spotify link for them.

Note: the client secret checked into `do_the_scrape.md` and the archive
notebooks is a live credential in a public repo and must not be used; it is
pending rotation.
"""

import json
import os
import pathlib

import config

CACHED_TRACKS_JSON = pathlib.Path.home() / "Projects/side-projects/sermons/tracks.json"


def from_cache() -> dict[str, str]:
    if not CACHED_TRACKS_JSON.exists():
        return {}
    tracks = json.loads(CACHED_TRACKS_JSON.read_text(encoding="utf-8"))
    return {
        str(t["soundcloud_id"]): t["spotify_id"]
        for t in tracks
        if t.get("soundcloud_id") and t.get("spotify_id")
    }


def from_api() -> dict[str, str]:
    """Match Spotify episodes to SoundCloud by title containment.

    Both feeds are published from the same source audio, so the SoundCloud title
    appears verbatim in the Spotify episode's name or description -- the same
    naive match the original `metadata.py` used, and it holds up.
    """
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not (client_id and client_secret):
        return {}

    import requests

    auth = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        timeout=15,
    )
    auth.raise_for_status()
    headers = {"Authorization": f"Bearer {auth.json()['access_token']}"}

    episodes = []
    url = f"https://api.spotify.com/v1/shows/{config.SPOTIFY_SHOW_ID}/episodes?market=BR&limit=50"
    while url:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        page = resp.json()
        episodes.extend(page["items"])
        url = page.get("next")

    # Keyed by SoundCloud id via the pending list, which carries both titles.
    pending = json.loads(config.PENDING_JSON.read_text(encoding="utf-8"))
    haystack = [(f"{e['name']} {e['description']}".lower(), e["id"]) for e in episodes if e]

    resolved = {}
    for track in pending:
        title = (track.get("title") or "").lower()
        if not title:
            continue
        hit = next((eid for text, eid in haystack if title in text), None)
        if hit:
            resolved[str(track["id"])] = hit
    return resolved


def load() -> dict[str, str]:
    ids = from_cache()
    print(f"spotify ids from cache: {len(ids)}", flush=True)
    try:
        api = from_api()
    except Exception as exc:
        print(f"spotify api lookup failed ({exc}); continuing without it", flush=True)
        api = {}
    if api:
        print(f"spotify ids from api: {len(api)}", flush=True)
        ids.update(api)  # live data wins over the stale scrape
    return ids
