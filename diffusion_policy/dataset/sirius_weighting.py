"""Utilities for Sirius-style intervention-aware loss weighting.

The functions in this module intentionally operate on frame indices and numpy
arrays only.  Keeping the label construction independent from image decoding
makes the weighting deterministic, cheap to audit, and straightforward to test.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


SIRIUS_DEMO = 0
SIRIUS_ROBOT = 1
SIRIUS_PRE_INTERVENTION = 2
SIRIUS_INTERVENTION = 3
SIRIUS_NUM_CLASSES = 4

SIRIUS_CLASS_NAMES = (
    "demo",
    "robot",
    "pre_intervention",
    "intervention",
)


@dataclass(frozen=True)
class SiriusTagAudit:
    changed_frames: int
    merged_policy_gaps: int
    removed_intervention_pulses: int
    segments_before: int
    segments_after: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "changed_frames": self.changed_frames,
            "merged_policy_gaps": self.merged_policy_gaps,
            "removed_intervention_pulses": self.removed_intervention_pulses,
            "segments_before": self.segments_before,
            "segments_after": self.segments_after,
        }


def _episode_ranges(episode_ends: Sequence[int]):
    start = 0
    for raw_end in episode_ends:
        end = int(raw_end)
        if end < start:
            raise ValueError("episode_ends must be monotonically increasing")
        yield start, end
        start = end


def _runs(values: np.ndarray):
    if values.ndim != 1:
        raise ValueError(f"Expected a 1-D array, got {values.shape}")
    if values.size == 0:
        return
    start = 0
    current = values[0]
    for idx in range(1, values.size):
        if values[idx] != current:
            yield start, idx, current
            start = idx
            current = values[idx]
    yield start, values.size, current


def _median_positive_step(timestamps: Optional[np.ndarray], fallback_fps: float) -> float:
    fallback = 1.0 / float(fallback_fps)
    if timestamps is None or len(timestamps) < 2:
        return fallback
    diffs = np.diff(timestamps.astype(np.float64, copy=False))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return fallback
    return float(np.median(diffs))


def _run_duration_seconds(
    start: int,
    end: int,
    timestamps: Optional[np.ndarray],
    fallback_fps: float,
) -> float:
    if end <= start:
        return 0.0
    if timestamps is None:
        return (end - start) / float(fallback_fps)
    step = _median_positive_step(timestamps, fallback_fps)
    return float(timestamps[end - 1] - timestamps[start] + step)


def _count_active_segments(tags: np.ndarray, episode_ends: Sequence[int]) -> int:
    count = 0
    for start, end in _episode_ranges(episode_ends):
        count += sum(int(value == 1) for _, _, value in _runs(tags[start:end]))
    return count


def clean_intervention_tags(
    hitl_tag: np.ndarray,
    episode_ends: Sequence[int],
    timestamps: Optional[np.ndarray] = None,
    merge_policy_gaps_seconds: float = 0.5,
    min_intervention_seconds: float = 0.3,
    fallback_fps: float = 15.0,
) -> Tuple[np.ndarray, SiriusTagAudit]:
    """Debounce binary takeover labels independently inside every episode.

    Short policy gaps bounded by takeover on both sides are closed first.  Then
    isolated takeover pulses shorter than ``min_intervention_seconds`` are
    removed.  The order preserves a real intervention that merely contains a
    momentary VR-tag dropout.
    """
    raw = (np.asarray(hitl_tag).reshape(-1) > 0).astype(np.uint8)
    ends = np.asarray(episode_ends, dtype=np.int64).reshape(-1)
    if ends.size == 0 or int(ends[-1]) != raw.size:
        raise ValueError(
            "episode_ends must terminate at the tag array length: "
            f"last_end={ends[-1] if ends.size else None}, length={raw.size}"
        )
    ts = None if timestamps is None else np.asarray(timestamps).reshape(-1)
    if ts is not None and ts.size != raw.size:
        raise ValueError("timestamps and hitl_tag must have the same length")
    if merge_policy_gaps_seconds < 0 or min_intervention_seconds < 0:
        raise ValueError("Sirius debounce durations must be non-negative")
    if fallback_fps <= 0:
        raise ValueError("fallback_fps must be positive")

    cleaned = raw.copy()
    segments_before = _count_active_segments(raw, ends)
    merged_gaps = 0
    removed_pulses = 0

    for ep_start, ep_end in _episode_ranges(ends):
        local_ts = None if ts is None else ts[ep_start:ep_end]
        local = cleaned[ep_start:ep_end]
        gap_runs = list(_runs(local.copy()))
        for run_idx, (start, end, value) in enumerate(gap_runs):
            bounded = run_idx > 0 and run_idx + 1 < len(gap_runs)
            if value != 0 or not bounded:
                continue
            if gap_runs[run_idx - 1][2] != 1 or gap_runs[run_idx + 1][2] != 1:
                continue
            duration = _run_duration_seconds(start, end, local_ts, fallback_fps)
            if duration <= merge_policy_gaps_seconds:
                local[start:end] = 1
                merged_gaps += 1

        pulse_runs = list(_runs(local.copy()))
        for start, end, value in pulse_runs:
            if value != 1:
                continue
            duration = _run_duration_seconds(start, end, local_ts, fallback_fps)
            if duration < min_intervention_seconds:
                local[start:end] = 0
                removed_pulses += 1

    audit = SiriusTagAudit(
        changed_frames=int(np.count_nonzero(cleaned != raw)),
        merged_policy_gaps=merged_gaps,
        removed_intervention_pulses=removed_pulses,
        segments_before=segments_before,
        segments_after=_count_active_segments(cleaned, ends),
    )
    return cleaned, audit


def build_online_frame_classes(
    cleaned_hitl_tag: np.ndarray,
    episode_ends: Sequence[int],
    timestamps: Optional[np.ndarray] = None,
    pre_intervention_seconds: float = 1.0,
    fallback_fps: float = 15.0,
) -> np.ndarray:
    """Assign ROBOT, PRE_INTERVENTION, or INTERVENTION to every online frame."""
    tags = (np.asarray(cleaned_hitl_tag).reshape(-1) > 0).astype(np.uint8)
    ends = np.asarray(episode_ends, dtype=np.int64).reshape(-1)
    if ends.size == 0 or int(ends[-1]) != tags.size:
        raise ValueError("episode_ends do not match cleaned_hitl_tag")
    ts = None if timestamps is None else np.asarray(timestamps).reshape(-1)
    if ts is not None and ts.size != tags.size:
        raise ValueError("timestamps and cleaned_hitl_tag must have the same length")
    if pre_intervention_seconds < 0:
        raise ValueError("pre_intervention_seconds must be non-negative")

    classes = np.full(tags.size, SIRIUS_ROBOT, dtype=np.int8)
    classes[tags == 1] = SIRIUS_INTERVENTION
    fallback_frames = int(np.ceil(pre_intervention_seconds * fallback_fps))

    for ep_start, ep_end in _episode_ranges(ends):
        local = tags[ep_start:ep_end]
        rising = np.flatnonzero((local == 1) & np.r_[True, local[:-1] == 0])
        for local_rise in rising:
            rise = ep_start + int(local_rise)
            if ts is None:
                pre_start = max(ep_start, rise - fallback_frames)
            else:
                threshold = float(ts[rise]) - pre_intervention_seconds
                pre_start = ep_start + int(
                    np.searchsorted(ts[ep_start:rise], threshold, side="left")
                )
            candidate = np.arange(pre_start, rise, dtype=np.int64)
            candidate = candidate[tags[candidate] == 0]
            classes[candidate] = SIRIUS_PRE_INTERVENTION
    return classes


def action_token_classes(
    frame_classes: np.ndarray,
    current_idx: int,
    horizon: int,
    stride: int,
    episode_end: int,
    action_padding: bool = False,
) -> np.ndarray:
    """Map an action chunk to per-token Sirius classes."""
    indices = int(current_idx) + np.arange(int(horizon), dtype=np.int64) * int(stride)
    if action_padding:
        indices = np.minimum(indices, int(episode_end) - 1)
    elif indices.size and indices[-1] >= int(episode_end):
        raise ValueError("Unpadded action token extends beyond its episode")
    if indices.size and (indices[0] < 0 or indices[-1] >= len(frame_classes)):
        raise IndexError("Action-token class index is outside the replay buffer")
    return np.asarray(frame_classes, dtype=np.int8)[indices]


def compute_sirius_importance_weights(
    buffer_class_counts: np.ndarray,
    buffer_sampling_weights: np.ndarray,
    target_intervention_fraction: float = 0.5,
    target_pre_intervention_fraction: float = 0.0,
    max_importance_weight: Optional[float] = None,
):
    """Compute proposal probabilities, target probabilities, and class weights.

    Remaining target mass is distributed between demo and robot in proportion
    to their proposal probabilities.  Counts must describe action-token
    occurrences in the same sampler domain used by training, not raw frames.
    """
    counts = np.asarray(buffer_class_counts, dtype=np.float64)
    sampling = np.asarray(buffer_sampling_weights, dtype=np.float64).reshape(-1)
    if counts.ndim != 2 or counts.shape[1] != SIRIUS_NUM_CLASSES:
        raise ValueError(
            f"buffer_class_counts must have shape [N,{SIRIUS_NUM_CLASSES}]"
        )
    if counts.shape[0] != sampling.size:
        raise ValueError("One sampling weight is required per buffer")
    if np.any(counts < 0) or np.any(sampling < 0) or sampling.sum() <= 0:
        raise ValueError("Counts and sampling weights must be non-negative")
    if not 0 <= target_intervention_fraction <= 1:
        raise ValueError("target_intervention_fraction must be in [0,1]")
    if not 0 <= target_pre_intervention_fraction <= 1:
        raise ValueError("target_pre_intervention_fraction must be in [0,1]")
    if target_intervention_fraction + target_pre_intervention_fraction > 1:
        raise ValueError("Sirius target fractions cannot sum above one")

    sampling = sampling / sampling.sum()
    per_buffer_total = counts.sum(axis=1)
    if np.any(per_buffer_total <= 0):
        raise ValueError("Every sampled buffer must contain at least one action token")
    per_buffer_prob = counts / per_buffer_total[:, None]
    proposal = (sampling[:, None] * per_buffer_prob).sum(axis=0)

    target = np.zeros(SIRIUS_NUM_CLASSES, dtype=np.float64)
    target[SIRIUS_INTERVENTION] = float(target_intervention_fraction)
    target[SIRIUS_PRE_INTERVENTION] = float(target_pre_intervention_fraction)
    remaining = 1.0 - target_intervention_fraction - target_pre_intervention_fraction
    positive_base_mass = proposal[SIRIUS_DEMO] + proposal[SIRIUS_ROBOT]
    if remaining > 0 and positive_base_mass <= 0:
        raise ValueError("No demo/robot proposal mass is available for the target")
    if positive_base_mass > 0:
        target[SIRIUS_DEMO] = remaining * proposal[SIRIUS_DEMO] / positive_base_mass
        target[SIRIUS_ROBOT] = remaining * proposal[SIRIUS_ROBOT] / positive_base_mass

    missing = (target > 0) & (proposal <= 0)
    if np.any(missing):
        names = [SIRIUS_CLASS_NAMES[i] for i in np.flatnonzero(missing)]
        raise ValueError(f"Positive Sirius target mass has no proposal samples: {names}")

    weights = np.zeros(SIRIUS_NUM_CLASSES, dtype=np.float64)
    present = proposal > 0
    weights[present] = target[present] / proposal[present]
    if max_importance_weight is not None:
        if max_importance_weight <= 0:
            raise ValueError("max_importance_weight must be positive when set")
        weights = np.minimum(weights, float(max_importance_weight))

    return proposal, target, weights
