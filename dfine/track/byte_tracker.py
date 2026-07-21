"""BYTETracker — the ByteTrack multi-object association algorithm.

A clean-room reimplementation of the standard ByteTrack pipeline (Zhang et al.,
*ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022; MIT):
Kalman motion prediction + a two-stage IoU association that matches high-score boxes
first, then recovers tracks from leftover low-score boxes. numpy + scipy only (torch-
free); scipy's ``linear_sum_assignment`` does the matching, so it is imported lazily.
"""

from __future__ import annotations

import numpy as np

from .kalman_filter import KalmanFilterXYAH

__all__ = ["BYTETracker", "STrack", "TrackState"]


class TrackState:
    """Lifecycle states for a track."""

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """A single tracked object: its Kalman state, score, class, and id."""

    _count = 0

    def __init__(self, tlwh, score, cls):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter: KalmanFilterXYAH | None = None
        self.mean = None
        self.covariance = None
        self.is_activated = False
        self.score = float(score)
        self.cls = cls
        self.tracklet_len = 0
        self.state = TrackState.New
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0

    @staticmethod
    def next_id() -> int:
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_id() -> None:
        STrack._count = 0

    def predict(self) -> None:
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def activate(self, kalman_filter: KalmanFilterXYAH, frame_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = kalman_filter.initiate(self._tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None:
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self._tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls

    def update(self, new_track: STrack, frame_id: int) -> None:
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self._tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    @property
    def tlwh(self) -> np.ndarray:
        """Current box as ``(top-left-x, top-left-y, w, h)``."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def _tlwh_to_xyah(tlwh) -> np.ndarray:
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def xyxy_to_tlwh(xyxy) -> np.ndarray:
        ret = np.asarray(xyxy, dtype=np.float32).copy()
        ret[2:] -= ret[:2]
        return ret


def _bbox_ious(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between two sets of ``xyxy`` boxes (shape ``(len(a), len(b))``)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _iou_distance(tracks_a, tracks_b) -> np.ndarray:
    """Cost matrix ``1 - IoU`` between two track/detection lists."""
    a = np.array([t.xyxy for t in tracks_a], dtype=np.float32).reshape(-1, 4)
    b = np.array([t.xyxy for t in tracks_b], dtype=np.float32).reshape(-1, 4)
    return 1.0 - _bbox_ious(a, b)


def _linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """Hungarian match on ``cost_matrix``, keeping only pairs with cost ``<= thresh``.

    Returns ``(matches, unmatched_a, unmatched_b)`` where ``matches`` is an ``(M, 2)``
    array of ``(row, col)`` indices.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(cost_matrix)
    matches = [[r, c] for r, c in zip(rows, cols) if cost_matrix[r, c] <= thresh]
    matched_a = {m[0] for m in matches}
    matched_b = {m[1] for m in matches}
    unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_a]
    unmatched_b = [i for i in range(cost_matrix.shape[1]) if i not in matched_b]
    matches = np.asarray(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)
    return matches, unmatched_a, unmatched_b


def _joint(a: list, b: list) -> list:
    seen = {t.track_id for t in a}
    return a + [t for t in b if t.track_id not in seen]


def _subtract(a: list, b: list) -> list:
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]


def _remove_duplicates(a: list, b: list):
    dist = _iou_distance(a, b)
    dup_a, dup_b = set(), set()
    for p, q in zip(*np.where(dist < 0.15)):
        age_p = a[p].frame_id - a[p].start_frame
        age_q = b[q].frame_id - b[q].start_frame
        (dup_b if age_p > age_q else dup_a).add(q if age_p > age_q else p)
    res_a = [t for i, t in enumerate(a) if i not in dup_a]
    res_b = [t for i, t in enumerate(b) if i not in dup_b]
    return res_a, res_b


class BYTETracker:
    """ByteTrack: Kalman prediction + two-stage IoU association.

    ``track_thresh`` splits detections into high/low score; ``match_thresh`` is the
    max IoU-distance for a valid first-stage match; ``track_buffer`` (scaled by
    ``frame_rate``) is how many frames a lost track survives before removal.
    """

    def __init__(
        self,
        track_thresh: float = 0.25,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: float = 30.0,
    ):
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.frame_id = 0
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.det_thresh = track_thresh + 0.1
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.kalman_filter = KalmanFilterXYAH()
        STrack.reset_id()

    def update(self, xyxy: np.ndarray, scores: np.ndarray, cls: np.ndarray) -> list[STrack]:
        """Advance one frame; return the currently active tracks."""
        self.frame_id += 1
        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        cls = np.asarray(cls).reshape(-1)

        activated, refind, lost, removed = [], [], [], []

        high = scores >= self.track_thresh
        low = (scores > 0.1) & (scores < self.track_thresh)
        dets = self._init_tracks(xyxy[high], scores[high], cls[high])
        dets_second = self._init_tracks(xyxy[low], scores[low], cls[low])

        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked = [t for t in self.tracked_stracks if t.is_activated]

        pool = _joint(tracked, self.lost_stracks)
        for t in pool:
            t.predict()
        matches, u_track, u_det = _linear_assignment(_iou_distance(pool, dets), self.match_thresh)
        for it, idet in matches:
            track, det = pool[it], dets[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind.append(track)

        r_tracked = [pool[i] for i in u_track if pool[i].state == TrackState.Tracked]
        matches, u_track2, _ = _linear_assignment(_iou_distance(r_tracked, dets_second), 0.5)
        for it, idet in matches:
            track, det = r_tracked[it], dets_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind.append(track)
        for it in u_track2:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        dets = [dets[i] for i in u_det]
        matches, u_unconfirmed, u_det = _linear_assignment(_iou_distance(unconfirmed, dets), 0.7)
        for it, idet in matches:
            unconfirmed[it].update(dets[idet], self.frame_id)
            activated.append(unconfirmed[it])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed.append(track)

        for idet in u_det:
            track = dets[idet]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = _joint(self.tracked_stracks, activated)
        self.tracked_stracks = _joint(self.tracked_stracks, refind)
        self.lost_stracks = _subtract(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = _subtract(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)
        self.tracked_stracks, self.lost_stracks = _remove_duplicates(
            self.tracked_stracks, self.lost_stracks
        )
        return [t for t in self.tracked_stracks if t.is_activated]

    @staticmethod
    def _init_tracks(xyxy, scores, cls) -> list[STrack]:
        if len(xyxy) == 0:
            return []
        return [STrack(STrack.xyxy_to_tlwh(b), s, c) for b, s, c in zip(xyxy, scores, cls)]
