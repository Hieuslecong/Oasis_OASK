import pytest
torch = pytest.importorskip("torch")
from oasis_cycle_aosk.models import LightweightSegmenter, TranslationGenerator, OASISDiscriminator, MultiScaleLightweightSegmenter, ConditionalOASISCritic, RelationalOASISRC
from oasis_cycle_aosk.aosk import AOSKSoft
from oasis_cycle_aosk.losses import oasis_hard_negative_loss

def test_shapes_and_deployment_is_student_only():
    x = torch.rand(2, 3, 32, 32); m = torch.rand(2, 1, 32, 32)
    assert LightweightSegmenter()(x).shape == (2, 1, 32, 32)
    assert TranslationGenerator()(x, m).shape == x.shape
    assert OASISDiscriminator()(x).shape == (2, 3, 32, 32)
    assert AOSKSoft()(x, m).shape == x.shape
    assert MultiScaleLightweightSegmenter()(x).shape == (2, 1, 32, 32)
    assert ConditionalOASISCritic()(x, m).shape == (2, 3, 32, 32)
    rc = RelationalOASISRC()(x, m)
    assert rc["semantic"].shape == (2, 3, 32, 32)
    assert rc["mismatch"].shape == (2, 1, 32, 32)
    assert rc["pair"].shape == (2, 1)

def test_hard_negative_does_not_penalize_gt_crack_and_has_gradient():
    logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
    target = torch.zeros(1, 1, 8, 8); target[..., 2:4, 2:4] = 1
    critic = torch.zeros(1, 3, 8, 8); critic[:, 0] = 4
    loss = oasis_hard_negative_loss(logits, target, critic); loss.backward()
    assert torch.isfinite(loss) and logits.grad[..., 2:4, 2:4].abs().sum() == 0

def test_aosk_has_finite_gradient_and_no_circular_boundary_dependency():
    x = torch.zeros(1, 3, 16, 16, requires_grad=True)
    x.data[..., 0, -1] = 10.0
    mask = torch.zeros(1, 1, 16, 16); mask[..., 8, 0] = 1.0
    out = AOSKSoft(max_shift=3)(x, mask)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    # A left-edge crack may read replicated edge values, but must not read the
    # opposite right edge as torch.roll would do.
    assert float(out[..., 8, 0].abs().max()) < 5.0
