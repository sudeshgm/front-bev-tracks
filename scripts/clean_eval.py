#!/usr/bin/env python3
"""Join split object ids in an existing eval.json.

No model is run. Lane curves are fitted in the overlay, so this does not
rewrite them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdtrack.tracks import merge_object_tracks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    doc = json.loads(args.eval.read_text())
    clip = json.loads(args.clip.read_text())
    token_time = {frame["token"]: frame["t"] for frame in clip["frames"]}
    before_objects = len(doc["object_tracks"])
    before_lanes = len(doc["lane_tracks"])
    doc["object_tracks"] = merge_object_tracks(doc["object_tracks"], token_time)
    doc["cleaned"] = {
        "object_tracks_before": before_objects,
        "lane_tracks_before": before_lanes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc))
    print(
        f"objects {before_objects} -> {len(doc['object_tracks'])}, "
        f"lanes {before_lanes} -> {len(doc['lane_tracks'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
