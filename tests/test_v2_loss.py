import pytest

torch = pytest.importorskip("torch")

from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.losses import (
    gt_anchored_relational_loss,
    oasis_rc_student_loss_v2,
)


def _relation(mismatch_logit):
    return {
        "mismatch": torch.full((1, 1, 1, 1), mismatch_logit),
        "pair": torch.zeros(1, 1),
    }


def test_v21_gt_energy_is_not_repelled_when_margin_feasible():
    e_pred = torch.tensor([0.10], requires_grad=True)
    e_gt = torch.tensor([0.10])
    e_corrupted = torch.tensor([0.80])
    loss, terms = gt_anchored_relational_loss(
        e_pred, e_gt, e_corrupted, margin=0.10
    )
    (gradient,) = torch.autograd.grad(loss, e_pred)
    assert float(loss) == pytest.approx(0.0, abs=1e-8)
    assert float(terms["anchor"]) == pytest.approx(0.0, abs=1e-8)
    assert float(terms["reject_corrupted"]) == pytest.approx(0.0, abs=1e-8)
    assert float(gradient) == pytest.approx(0.0, abs=1e-8)


def test_v21_anchor_pulls_prediction_back_to_gt_energy():
    e_pred = torch.tensor([0.40], requires_grad=True)
    e_gt = torch.tensor([0.10])
    e_corrupted = torch.tensor([1.00])
    loss, _ = gt_anchored_relational_loss(
        e_pred, e_gt, e_corrupted, margin=0.10
    )
    (gradient,) = torch.autograd.grad(loss, e_pred)
    assert float(gradient) > 0.0  # gradient descent lowers e_pred toward e_gt


def test_v21_rejection_pushes_prediction_below_corruption_margin():
    e_pred = torch.tensor([0.95], requires_grad=True)
    e_gt = torch.tensor([0.10])
    e_corrupted = torch.tensor([1.00])
    loss, terms = gt_anchored_relational_loss(
        e_pred, e_gt, e_corrupted, margin=0.10
    )
    (gradient,) = torch.autograd.grad(loss, e_pred)
    assert float(terms["reject_corrupted"]) > 0.0
    assert float(gradient) > 0.0  # gradient descent lowers e_pred


def test_v21_gt_and_corruption_energies_are_detached():
    e_pred = torch.tensor([0.5], requires_grad=True)
    e_gt = torch.tensor([0.1], requires_grad=True)
    e_corrupted = torch.tensor([0.9], requires_grad=True)
    loss, _ = gt_anchored_relational_loss(e_pred, e_gt, e_corrupted, margin=0.1)
    loss.backward()
    assert e_pred.grad is not None
    assert e_gt.grad is None
    assert e_corrupted.grad is None


def test_v21_student_loss_has_student_gradient_only():
    critic = OASISRCv2Critic(width=4)
    for parameter in critic.parameters():
        parameter.requires_grad_(False)

    x = torch.rand(2, 3, 32, 32)
    y = torch.zeros(2, 1, 32, 32)
    y[:, :, 8:24, 15:17] = 1
    logits = torch.zeros(2, 1, 32, 32, requires_grad=True)
    prediction = logits.sigmoid()
    corrupted = y.flip(-1)

    with torch.no_grad():
        gt_out = critic(x, y)
        corrupted_out = critic(x, corrupted)

    loss, terms = oasis_rc_student_loss_v2(
        critic(x, prediction),
        gt_out,
        corrupted_out,
        prediction,
        y,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in critic.parameters())
    assert {
        "anchor",
        "reject_corrupted",
        "fp",
        "e_pred",
        "e_gt",
        "e_corrupted",
        "energy_gap",
        "separated_fraction",
    }.issubset(terms)


def test_v21_critic_output_contract():
    out = OASISRCv2Critic(width=4)(
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 1, 32, 32),
    )
    assert out["semantic"].shape == (1, 3, 32, 32)
    assert out["mismatch"].shape == (1, 1, 32, 32)
    assert out["pair"].shape == (1, 1)
