"""The merged DP must retain both Sirius loss weights and residual features."""
import types

import numpy as np
import torch
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def output_shape(self):
        return (4,)

    def forward(self, obs):
        self.calls += 1
        return obs["feature"]


class Prediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(()))

    def forward(self, sample, *args, **kwargs):
        return self.value.expand_as(sample)


def identity(width):
    ones = np.ones(width, dtype=np.float32)
    zeros = np.zeros(width, dtype=np.float32)
    return SingleFieldLinearNormalizer.create_manual(
        scale=ones, offset=zeros,
        input_stats_dict={"min": -ones, "max": ones, "mean": zeros, "std": ones},
    )


def make_policy():
    policy = DiffusionUnetTimmPolicy(
        shape_meta={"action": {"shape": [10], "horizon": 4}},
        noise_scheduler=DDPMScheduler(num_train_timesteps=4, prediction_type="sample"),
        obs_encoder=Encoder(), down_dims=(8,), diffusion_step_embed_dim=8,
        n_groups=4, input_pertub=0.0, train_diffusion_n_samples=2,
    )
    policy.model = Prediction()
    policy.normalizer["action"] = identity(10)
    policy.normalizer["feature"] = identity(4)
    return policy


def test_sirius_weighted_masked_loss_backpropagates_after_residual_merge():
    policy = make_policy()
    actions = torch.ones(1, 4, 10)
    actions[:, 2:] = 3.0
    batch = {"obs": {"feature": torch.zeros(1, 4)}, "action": actions,
             "action_loss_weight": torch.tensor([[1.0, 3.0, 0.0, 0.0]])}
    loss = policy.compute_loss(batch)
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    torch.testing.assert_close(policy.model.value.grad, torch.tensor(-2.0))
    policy.model.value.grad = None
    batch["action_valid_mask"] = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    masked_loss = policy.compute_loss(batch)
    torch.testing.assert_close(masked_loss, torch.tensor(0.0))


def test_residual_receives_features_from_the_same_base_sample():
    policy = make_policy()
    seen = {}

    def sample(self, condition_data, condition_mask, global_cond, **kwargs):
        seen["features"] = global_cond
        return torch.full_like(condition_data, 0.25)

    policy.conditional_sample = types.MethodType(sample, policy)
    observation = {"feature": torch.randn(2, 4)}
    result = policy.predict_action(observation, return_obs_features=True)
    assert policy.obs_encoder.calls == 1
    assert result["obs_features"] is seen["features"]
    torch.testing.assert_close(result["action"], torch.full((2, 4, 10), 0.25))
    assert set(policy.predict_action(observation)) == {"action", "action_pred"}
