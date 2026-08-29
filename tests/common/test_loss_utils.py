import torch

from diffusion_policy.common.loss_utils import reduce_action_loss


def test_reduce_action_loss_matches_legacy_mean_without_weights():
    loss = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    assert torch.equal(reduce_action_loss(loss), loss.mean())


def test_reduce_action_loss_combines_valid_mask_and_token_weight():
    loss = torch.tensor(
        [
            [[1.0, 3.0], [10.0, 14.0], [100.0, 200.0]],
            [[2.0, 4.0], [20.0, 24.0], [300.0, 400.0]],
        ]
    )
    valid = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    weight = torch.tensor([[1.0, 2.0, 9.0], [3.0, 8.0, 7.0]])

    actual = reduce_action_loss(
        loss,
        action_valid_mask=valid,
        action_loss_weight=weight,
    )
    expected = (
        (1.0 + 3.0) * 1.0
        + (10.0 + 14.0) * 2.0
        + (2.0 + 4.0) * 3.0
    ) / (2 * (1.0 + 2.0 + 3.0))
    assert torch.isclose(actual, torch.tensor(expected))


def test_zero_weight_tokens_do_not_contribute():
    loss = torch.tensor([[[2.0], [999.0]]])
    weight = torch.tensor([[1.0, 0.0]])
    assert torch.equal(reduce_action_loss(loss, action_loss_weight=weight), loss[0, 0, 0])
