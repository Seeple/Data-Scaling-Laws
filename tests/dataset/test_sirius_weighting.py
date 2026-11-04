import numpy as np

from diffusion_policy.dataset.sirius_weighting import (
    SIRIUS_DEMO,
    SIRIUS_INTERVENTION,
    SIRIUS_PRE_INTERVENTION,
    SIRIUS_ROBOT,
    action_token_classes,
    build_online_frame_classes,
    clean_intervention_tags,
    compute_sirius_importance_weights,
)


def test_debounce_is_episode_local_and_closes_short_policy_gap():
    tags = np.array(
        [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=np.uint8
    )
    episode_ends = np.array([6, 12])
    timestamps = np.arange(len(tags), dtype=np.float64) / 10.0

    cleaned, audit = clean_intervention_tags(
        tags,
        episode_ends,
        timestamps=timestamps,
        merge_policy_gaps_seconds=0.11,
        min_intervention_seconds=0.15,
        fallback_fps=10.0,
    )

    # Episode 0's one-frame dropout is closed. Episode 1's isolated pulse is
    # removed and is never joined to episode 0 across the boundary.
    np.testing.assert_array_equal(
        cleaned,
        np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8),
    )
    assert audit.merged_policy_gaps == 1
    assert audit.removed_intervention_pulses == 1


def test_pre_intervention_window_uses_timestamps_and_never_crosses_episode():
    tags = np.array([0, 0, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
    ends = np.array([5, 8])
    timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 10.0, 10.1, 10.2])

    classes = build_online_frame_classes(
        tags,
        ends,
        timestamps=timestamps,
        pre_intervention_seconds=0.2,
        fallback_fps=10.0,
    )

    np.testing.assert_array_equal(
        classes,
        np.array(
            [
                SIRIUS_ROBOT,
                SIRIUS_PRE_INTERVENTION,
                SIRIUS_PRE_INTERVENTION,
                SIRIUS_INTERVENTION,
                SIRIUS_INTERVENTION,
                SIRIUS_PRE_INTERVENTION,
                SIRIUS_PRE_INTERVENTION,
                SIRIUS_INTERVENTION,
            ],
            dtype=np.int8,
        ),
    )


def test_action_chunk_gets_per_token_classes_at_stride_three():
    frame_classes = np.full(20, SIRIUS_ROBOT, dtype=np.int8)
    frame_classes[6:9] = SIRIUS_PRE_INTERVENTION
    frame_classes[9:] = SIRIUS_INTERVENTION

    classes = action_token_classes(
        frame_classes,
        current_idx=0,
        horizon=5,
        stride=3,
        episode_end=20,
    )

    np.testing.assert_array_equal(
        classes,
        [
            SIRIUS_ROBOT,
            SIRIUS_ROBOT,
            SIRIUS_PRE_INTERVENTION,
            SIRIUS_INTERVENTION,
            SIRIUS_INTERVENTION,
        ],
    )


def test_importance_weights_recover_requested_target_mass():
    counts = np.array(
        [
            [1000, 0, 0, 0],
            [0, 600, 100, 300],
        ],
        dtype=np.float64,
    )
    proposal, target, weights = compute_sirius_importance_weights(
        counts,
        buffer_sampling_weights=np.array([0.5, 0.5]),
        target_intervention_fraction=0.5,
        target_pre_intervention_fraction=0.0,
    )

    np.testing.assert_allclose(proposal * weights, target)
    assert target[SIRIUS_INTERVENTION] == 0.5
    assert target[SIRIUS_PRE_INTERVENTION] == 0.0
    assert weights[SIRIUS_PRE_INTERVENTION] == 0.0
    assert target[SIRIUS_DEMO] + target[SIRIUS_ROBOT] == 0.5
