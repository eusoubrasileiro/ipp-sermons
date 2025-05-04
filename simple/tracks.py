#!/usr/bin/env python3
"""
scrape_soundcloud_tracks.py
Generate tracks.json for https://soundcloud.com/ipperegrinos
▸ pip install yt-dlp
"""

import json
from pathlib import Path
from datetime import datetime
import yt_dlp                           # yt‑dlp 2024.10.13 or newer

PLAYLIST_URL = "https://soundcloud.com/ipperegrinos"
OUTFILE      = Path("tracks.json")

ydl_opts = {
    "extract_flat": True,   # one JSON object per track, no media download
    "skip_download": True,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(PLAYLIST_URL, download=False)

tracks = []
for entry in info["entries"]:
    tracks.append({
        "id"          : entry["id"],
        "title"       : entry["title"],
        "url"         : entry["url"],        # direct SoundCloud permalink
        "duration"    : entry.get("duration"),          # seconds (int) or None
        "upload_date" : (
            datetime.strptime(entry["upload_date"], "%Y%m%d")
                     .strftime("%Y-%m-%d")   # nicer format
            if entry.get("upload_date") else None
        ),
    })

OUTFILE.write_text(json.dumps(tracks, ensure_ascii=False, indent=2))
print(f"Wrote {len(tracks)} tracks → {OUTFILE.resolve()}")
