import pytest

torch = pytest.importorskip("torch")

from oasis_cycle_aosk.topology_loss import centerline_cldice_loss


def test_topology_loss_has_gradient_and_empty_mask_is_safe():
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 8:24, 15:17] = 1
    loss = centerline_cldice_loss(logits, target, iterations=4)
    assert torch.isfinite(loss)
    loss.backward()
    assert float(logits.grad[0].abs().sum()) > 0

    empty_logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    empty_loss = centerline_cldice_loss(
        empty_logits, torch.zeros_like(empty_logits), iterations=2
    )
    assert float(empty_loss.detach()) == 0.0
    empty_loss.backward()
    assert float(empty_logits.grad.abs().sum()) == 0.0
