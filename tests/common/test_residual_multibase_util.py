import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.common.residual_multibase_util import (
    MultiBaseFilterConfig,
    evaluate_multibase_candidate,
)


def _chunk():
    chunk = np.zeros((16, 8), dtype=np.float64)
    chunk[:, 0] = np.linspace(0.4, 0.5, 16)
    chunk[:, 3:7] = Rotation.identity().as_quat()
    chunk[:, 7] = 0.04
    return chunk


def test_filtered_multibase_accepts_compatible_diverse_candidate():
    recorded = _chunk()
    edited = recorded.copy()
    edited[:11, 1] += 0.02
    candidate = recorded.copy()
    candidate[:5, 0] += 0.003
    mask = np.zeros(16, dtype=bool)
    mask[:11] = True
    accepted, reason, metrics = evaluate_multibase_candidate(
        candidate,
        recorded,
        edited,
        mask,
        [recorded],
        MultiBaseFilterConfig(),
    )
    assert accepted
    assert reason == "accepted"
    assert metrics["max_target_position_residual_m"] < 0.1


def test_filtered_multibase_rejects_incompatible_and_duplicate_candidates():
    recorded = _chunk()
    edited = recorded.copy()
    mask = np.zeros(16, dtype=bool)
    mask[:11] = True
    config = MultiBaseFilterConfig()

    far = recorded.copy()
    far[:5, 0] += 0.2
    accepted, reason, _ = evaluate_multibase_candidate(
        far, recorded, edited, mask, [recorded], config
    )
    assert not accepted
    assert reason == "first_position_deviation"

    accepted, reason, _ = evaluate_multibase_candidate(
        recorded.copy(), recorded, edited, mask, [recorded], config
    )
    assert not accepted
    assert reason == "insufficient_diversity"
