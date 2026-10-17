"""Display-only pixel tracks; no predicted heads or future reference labels."""

from collections import Counter
from dataclasses import asdict

import numpy as np

from click2future.evaluation.tracking import TRACKER_VERSION, TrackerConfig, track_video


def pixel_diagnostics(videos, initial_masks, object_valid):
    settings = TrackerConfig()
    result = {
        "source": "generated pixels",
        "extractor_version": TRACKER_VERSION,
        "settings": asdict(settings),
        "adoption": "candidate pending independent calibration review",
        "samples": [],
    }
    try:
        for video in videos:
            tracks = track_video(video, initial_masks, object_valid, settings)
            # Keep gaps explicit. The browser draws only contiguous accepted segments.
            positions = tracks.positions.astype(object)
            positions[~tracks.tracked] = None
            reasons = tracks.reasons[1:, object_valid].ravel().tolist()
            result["samples"].append(
                {
                    "positions": positions.tolist(),
                    "tracked": tracks.tracked.tolist(),
                    "future_tracked": int(tracks.tracked[1:, object_valid].sum()),
                    "future_expected": int(object_valid.sum()) * (len(video) - 1),
                    "reasons": dict(Counter(reasons)),
                }
            )
    except ValueError as error:
        return {**result, "status": "unavailable", "reason": str(error), "samples": []}
    return {**result, "status": "available", "object_ids": np.flatnonzero(object_valid).tolist()}
