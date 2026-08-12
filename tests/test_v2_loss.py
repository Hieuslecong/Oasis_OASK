import pytest

torch = pytest.importorskip("torch")

from oasis_cycle_aosk.losses_v2 import oasis_rc_student_loss_v2
from oasis_cycle_aosk.models import RelationalOASISRC


def test_v2_corrupted_ranking_has_student_gradient_only():
    critic = RelationalOASISRC(width=4)
    for parameter in critic.parameters():
        parameter.requires_grad_(False)

    image = torch.rand(2, 3, 32, 32)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:24, 15:17] = 1.0
    student_logits = torch.zeros(2, 1, 32, 32, requires_grad=True)
    student_mask = student_logits.sigmoid()
    corrupted = target.flip(-1)

    with torch.no_grad():
        gt_out = critic(image, target)
        corrupted_out = critic(image, corrupted)
    pred_out = critic(image, student_mask)
    loss, terms = oasis_rc_student_loss_v2(
        pred_out, gt_out, corrupted_out, student_mask, target
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(student_logits.grad).all()
    assert student_logits.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in critic.parameters())
    assert set(terms) == {"rank_gt", "rank_corrupted", "fp"}


def test_v2_critic_output_contract():
    critic = RelationalOASISRC(width=4)
    out = critic(torch.rand(1, 3, 32, 32), torch.rand(1, 1, 32, 32))
    assert out["semantic"].shape == (1, 3, 32, 32)
    assert out["mismatch"].shape == (1, 1, 32, 32)
    assert out["pair"].shape == (1, 1)
