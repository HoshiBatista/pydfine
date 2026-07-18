"""Multi-object tracking for DFINE detections.

:class:`ByteTrack` wraps the vendored :class:`~dfine.track.byte_tracker.BYTETracker`
so per-frame :class:`~dfine.results.Results` gain a persistent track id per box across
a video. Used by :meth:`dfine.DFINE.predict_video` with ``track=True``; the core is a
clean-room port of ByteTrack (Zhang et al., ECCV 2022; MIT) and depends only on
numpy + scipy (no torch, no ``supervision``).
"""

from __future__ import annotations

import numpy as np

from ..results import Boxes, Results
from .byte_tracker import BYTETracker, STrack, TrackState

__all__ = ["ByteTrack", "BYTETracker", "STrack", "TrackState"]


class ByteTrack:
    """Stateful tracker: feed it one :class:`Results` per frame, get ids back.

    Construct one per video (state persists across :meth:`update` calls), then call
    :meth:`update` on each frame's detections. The returned :class:`Results` carries
    only the currently active tracks, each with ``boxes.id`` set to its track id.
    """

    def __init__(
        self,
        frame_rate: float = 30.0,
        track_thresh: float = 0.25,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
    ):
        self._tracker = BYTETracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
        )

    def update(self, result: Results) -> Results:
        """Track ``result``'s detections into the running tracks; return tracked boxes."""
        import torch

        boxes = result.boxes
        if len(boxes):
            xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32).reshape(-1, 4)
            scores = boxes.conf.detach().cpu().numpy().astype(np.float32).reshape(-1)
            cls = boxes.cls.detach().cpu().numpy().reshape(-1)
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.float32)

        tracks = self._tracker.update(xyxy, scores, cls)
        if tracks:
            t_xyxy = np.stack([t.xyxy for t in tracks]).astype(np.float32)
            t_conf = np.array([t.score for t in tracks], dtype=np.float32)
            t_cls = np.array([int(t.cls) for t in tracks], dtype=np.int64)
            t_id = np.array([int(t.track_id) for t in tracks], dtype=np.int64)
        else:
            t_xyxy = np.zeros((0, 4), dtype=np.float32)
            t_conf = np.zeros((0,), dtype=np.float32)
            t_cls = np.zeros((0,), dtype=np.int64)
            t_id = np.zeros((0,), dtype=np.int64)

        tracked = Boxes(
            xyxy=torch.from_numpy(t_xyxy),
            conf=torch.from_numpy(t_conf),
            cls=torch.from_numpy(t_cls),
            id=torch.from_numpy(t_id),
        )
        return Results(result.orig_img, tracked, result.names)
