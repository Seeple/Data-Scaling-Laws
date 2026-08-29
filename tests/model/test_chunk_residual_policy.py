import numpy as np
import torch

from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
from diffusion_policy.model.residual.chunk_residual_mlp import ChunkResidualMLP
from diffusion_policy.policy.chunk_residual_policy import ChunkResidualPolicy


def _identity_normalizer():
    ones = np.ones(7, dtype=np.float32)
    zeros = np.zeros(7, dtype=np.float32)
    return SingleFieldLinearNormalizer.create_manual(
        scale=ones,
        offset=zeros,
        input_stats_dict={
            "min": -ones,
            "max": ones,
            "mean": zeros,
            "std": ones,
        },
    )


def test_zero_initialized_model_is_exact_base_identity():
    model = ChunkResidualMLP(obs_feature_dim=32, gate_enabled=False)
    policy = ChunkResidualPolicy(model)
    policy.set_residual_normalizer(_identity_normalizer())
    output = policy.predict_residual(
        torch.randn(3, 32), torch.randn(3, 16, 10)
    )
    assert torch.count_nonzero(output["residual"]) == 0
    assert torch.count_nonzero(output["applied_residual"]) == 0


def test_masked_correction_zero_and_gate_loss_backpropagate():
    model = ChunkResidualMLP(obs_feature_dim=32, gate_enabled=True)
    policy = ChunkResidualPolicy(model)
    policy.set_residual_normalizer(_identity_normalizer())
    target = torch.randn(4, 16, 7) * 0.1
    valid = torch.ones(4, 16, dtype=torch.bool)
    valid[0, :5] = False
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    output = policy(
        torch.randn(4, 32),
        torch.randn(4, 16, 10),
        target,
        valid,
        labels,
    )
    assert torch.isfinite(output["loss"])
    assert output["correction_loss"] > 0
    assert output["zero_loss"] > 0
    assert output["gate_loss"] > 0
    output["loss"].backward()
    assert model.residual_head.weight.grad is not None
    assert model.gate_head.weight.grad is not None


def test_explicit_step_horizon_is_independent_of_frozen_dp_horizon():
    model = ChunkResidualMLP(
        obs_feature_dim=32,
        action_horizon=None,
        base_action_horizon=5,
        residual_horizon=5,
        gate_enabled=False,
    )
    policy = ChunkResidualPolicy(model)
    policy.set_residual_normalizer(_identity_normalizer())
    output = policy.predict_residual(
        torch.randn(2, 32), torch.randn(2, 5, 10)
    )
    assert policy.base_action_horizon == 5
    assert policy.residual_horizon == 5
    assert output["residual"].shape == (2, 5, 7)


def test_nonhuman_rollout_target_can_supervise_residual_and_gate():
    """Round-two rollout targets need not pose as human correction labels."""
    model = ChunkResidualMLP(obs_feature_dim=32, gate_enabled=True)
    policy = ChunkResidualPolicy(model)
    policy.set_residual_normalizer(_identity_normalizer())
    output = policy(
        torch.zeros(1, 32),
        torch.zeros(1, 16, 10),
        torch.ones(1, 16, 7) * 0.1,
        torch.ones(1, 16, dtype=torch.bool),
        correction_label=torch.zeros(1),
        residual_supervision_label=torch.ones(1),
        gate_target=torch.ones(1),
    )
    assert output["correction_loss"] > 0
    assert output["zero_loss"] == 0
    assert output["gate_loss"] > 0
