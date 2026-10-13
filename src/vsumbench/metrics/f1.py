"""F1 between a machine summary and each ground-truth user summary.

Matches the lineage ``evaluate_summary`` (KaiyangZhou/pytorch-vsumm-reinforce
and e-apostolidis/PGL-SUM): per-user precision ``overlap / |summary|``,
recall ``overlap / |user summary|``, harmonic mean. Two deliberate,
documented deviations from individual lineage variants
(docs/protocols/eccv16-summary-protocol.md): the score is a ratio in
``[0, 1]`` (PGL-SUM multiplies by 100), and an empty machine summary or
empty user summary yields 0 instead of the DR-DSN ``1e-8`` epsilon / the
PGL-SUM ZeroDivisionError.
"""

from __future__ import annotations

import numpy as np

from vsumbench.types import Mask, Video

__all__ = ["F1_METRIC", "F1Metric"]


class F1Metric:
    """Binary frame-level F1, one value per user summary."""

    name = "f1"

    def score(self, summary: Mask, video: Video) -> np.ndarray:
        if summary.video_id != video.video_id:
            raise ValueError(
                f"summary id {summary.video_id!r} does not match video id {video.video_id!r}"
            )
        if summary.frames.shape[0] != video.n_frames:
            raise ValueError(
                f"summary has {summary.frames.shape[0]} frames but video "
                f"{video.video_id!r} has {video.n_frames}"
            )
        if not video.user_summaries:
            raise ValueError(f"video {video.video_id!r} carries no user_summaries; F1 is undefined")
        machine_count = int(np.count_nonzero(summary.frames))
        scores = np.empty(len(video.user_summaries), dtype=np.float64)
        for i, user_summary in enumerate(video.user_summaries):
            overlap = int(np.count_nonzero(summary.frames & user_summary))
            user_count = int(np.count_nonzero(user_summary))
            precision = overlap / machine_count if machine_count else 0.0
            recall = overlap / user_count if user_count else 0.0
            if precision + recall == 0.0:
                scores[i] = 0.0
            else:
                scores[i] = 2.0 * precision * recall / (precision + recall)
        return scores


F1_METRIC = F1Metric()
