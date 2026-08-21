import pytest

torch = pytest.importorskip("torch")

from oasis_cycle_aosk.aosk import orientation_weights, oriented_consistency_loss


def test_flat_region_orientation_is_isotropic():
    ex = torch.zeros(1, 1, 4, 4)
    ey = torch.zeros_like(ex)
    wx, wy = orientation_weights(ex, ey)
    assert torch.allclose(wx, torch.full_like(wx, 0.5))
    assert torch.allclose(wy, torch.full_like(wy, 0.5))


def test_orientation_weights_sum_to_one_for_nonflat_regions():
    ex = torch.full((1, 1, 2, 2), 2.0)
    ey = torch.full((1, 1, 2, 2), 1.0)
    wx, wy = orientation_weights(ex, ey)
    assert torch.allclose(wx + wy, torch.ones_like(wx), atol=1e-6)


def test_oriented_consistency_is_finite_on_flat_image():
    image = torch.zeros(1, 3, 8, 8)
    logits = torch.randn(1, 1, 8, 8, requires_grad=True)
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, 2:6, 3:5] = 1
    loss = oriented_consistency_loss(logits, image, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
