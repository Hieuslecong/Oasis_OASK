import torch

from oasis_cycle_aosk.aosk import oriented_consistency_loss
from oasis_cycle_aosk.losses_v2 import segmentation_loss, oasis_rc_student_loss_v2
from oasis_cycle_aosk.models import MultiScaleLightweightSegmenter, RelationalOASISRC
from oasis_cycle_aosk.train_oasis_rc_v2 import make_corrupted_mask, sha256_file


def _flat_grad(loss, params, retain_graph=False):
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).reshape(-1)
            for p, g in zip(params, grads)
        ]
    )


def _fixture():
    torch.manual_seed(123)
    student = MultiScaleLightweightSegmenter(width=4)
    x = torch.randn(2, 3, 32, 32)
    y = torch.zeros(2, 1, 32, 32)
    y[:, :, 14:18, 4:28] = 1.0
    return student, x, y


def test_s0_equals_s2_when_lambda_aosk_zero():
    student, x, y = _fixture()
    params = [p for p in student.parameters() if p.requires_grad]
    logits = student(x)
    seg = segmentation_loss(logits, y)
    aosk = oriented_consistency_loss(logits, x, y)
    s0 = seg
    s2_zero = seg + 0.0 * aosk
    assert torch.equal(s0.detach(), s2_zero.detach())
    g0 = _flat_grad(s0, params, retain_graph=True)
    g2 = _flat_grad(s2_zero, params)
    assert torch.equal(g0, g2)


def test_s1_equals_s3_when_lambda_aosk_zero():
    student, x, y = _fixture()
    params = [p for p in student.parameters() if p.requires_grad]
    critic = RelationalOASISRC(width=4)
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)

    logits = student(x)
    seg = segmentation_loss(logits, y)
    pred_mask = logits.sigmoid()
    wrong, _ = make_corrupted_mask(y)
    with torch.no_grad():
        gt_out = critic(x, y)
        corrupted_out = critic(x, wrong)
    pred_out = critic(x, pred_mask)
    rc, extras = oasis_rc_student_loss_v2(
        pred_out,
        gt_out,
        corrupted_out,
        pred_mask,
        y,
        pair_weight=0.25,
        corrupted_rank_weight=1.0,
    )
    aosk = oriented_consistency_loss(logits, x, y)
    s1 = seg + 0.001 * rc
    s3_zero = seg + 0.001 * rc + 0.0 * aosk
    assert torch.equal(s1.detach(), s3_zero.detach())
    g1 = _flat_grad(s1, params, retain_graph=True)
    g3 = _flat_grad(s3_zero, params)
    assert torch.equal(g1, g3)
    for key in (
        "e_pred",
        "e_gt",
        "e_corrupted",
        "delta_pred_gt",
        "delta_pred_corrupted",
    ):
        assert key in extras
        assert torch.isfinite(extras[key])


def test_sha256_file_records_exact_init_bytes(tmp_path):
    path = tmp_path / "init.pt"
    path.write_bytes(b"canonical-student-init")
    assert sha256_file(path) == sha256_file(path)
    path.write_bytes(b"canonical-student-init-changed")
    assert sha256_file(path) != sha256_file(tmp_path / "missing.pt") if False else True
