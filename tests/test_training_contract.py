import copy
import hashlib

import torch

from oasis_cycle_aosk.aosk import oriented_consistency_loss
from oasis_cycle_aosk.losses_v2 import segmentation_loss, oasis_rc_student_loss_v2
from oasis_cycle_aosk.models import MultiScaleLightweightSegmenter, RelationalOASISRC
from oasis_cycle_aosk.train_oasis_rc_v2 import (
    augment,
    build_targets,
    make_corrupted_mask,
    make_generator,
    sha256_file,
)


def _flat_grad(loss, params, retain_graph=False):
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, allow_unused=True
    )
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


def test_rc_corruption_rng_does_not_change_augmentation_sequence():
    x = torch.linspace(-1, 1, 2 * 3 * 16 * 16).reshape(2, 3, 16, 16)
    y = torch.zeros(2, 1, 16, 16)
    y[:, :, 6:10, 3:13] = 1.0

    control_aug = make_generator(torch.device("cpu"), 777)
    connected_aug = make_generator(torch.device("cpu"), 777)
    corruption_gen = make_generator(torch.device("cpu"), 999)

    c1 = augment(x.clone(), y.clone(), generator=control_aug)
    c2 = augment(x.clone(), y.clone(), generator=control_aug)

    r1 = augment(x.clone(), y.clone(), generator=connected_aug)
    make_corrupted_mask(y, generator=corruption_gen)
    r2 = augment(x.clone(), y.clone(), generator=connected_aug)

    assert torch.equal(c1[0], r1[0])
    assert torch.equal(c1[1], r1[1])
    assert torch.equal(c2[0], r2[0])
    assert torch.equal(c2[1], r2[1])


def test_noop_corruption_is_pair_valid_not_forced_invalid():
    mask = torch.zeros(2, 1, 8, 8)
    invalid = torch.zeros_like(mask)
    _, mismatch, pair_valid = build_targets(mask, invalid)
    assert float(mismatch.sum()) == 0.0
    assert torch.equal(pair_valid, torch.ones_like(pair_valid))


def test_two_step_control_matches_connected_when_rc_weight_zero():
    base, x, y = _fixture()
    control = copy.deepcopy(base)
    connected = copy.deepcopy(base)
    critic = RelationalOASISRC(width=4).eval()
    for p in critic.parameters():
        p.requires_grad_(False)

    opt_c = torch.optim.AdamW(control.parameters(), lr=1e-4)
    opt_r = torch.optim.AdamW(connected.parameters(), lr=1e-4)
    aug_c = make_generator(torch.device("cpu"), 101)
    aug_r = make_generator(torch.device("cpu"), 101)
    rc_gen = make_generator(torch.device("cpu"), 202)

    for _ in range(2):
        xc, yc = augment(x.clone(), y.clone(), generator=aug_c)
        xr, yr = augment(x.clone(), y.clone(), generator=aug_r)
        assert torch.equal(xc, xr)
        assert torch.equal(yc, yr)

        lc = segmentation_loss(control(xc), yc)
        opt_c.zero_grad()
        lc.backward()
        opt_c.step()

        logits = connected(xr)
        seg = segmentation_loss(logits, yr)
        pred = logits.sigmoid()
        wrong, _ = make_corrupted_mask(yr, generator=rc_gen)
        with torch.no_grad():
            gt_out = critic(xr, yr)
            corrupt_out = critic(xr, wrong)
        pred_out = critic(xr, pred)
        rc, _ = oasis_rc_student_loss_v2(
            pred_out, gt_out, corrupt_out, pred, yr
        )
        lr = seg + 0.0 * rc
        opt_r.zero_grad()
        lr.backward()
        opt_r.step()

    for (name_c, tensor_c), (name_r, tensor_r) in zip(
        control.state_dict().items(), connected.state_dict().items()
    ):
        assert name_c == name_r
        assert torch.equal(tensor_c, tensor_r), name_c


def test_sha256_file_records_exact_init_bytes(tmp_path):
    path = tmp_path / "init.pt"
    first = b"canonical-student-init"
    second = b"canonical-student-init-changed"
    path.write_bytes(first)
    first_sha = sha256_file(path)
    assert first_sha == hashlib.sha256(first).hexdigest()
    path.write_bytes(second)
    second_sha = sha256_file(path)
    assert second_sha == hashlib.sha256(second).hexdigest()
    assert first_sha != second_sha
