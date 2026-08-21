import copy
import hashlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from oasis_cycle_aosk.aosk import oriented_consistency_loss
from oasis_cycle_aosk.models import MultiScaleLightweightSegmenter
from oasis_cycle_aosk.train_oasis_rc_v2 import (
    augment,
    build_targets,
    load_student_init,
    make_corrupted_mask,
    make_generator,
    sha256_file,
    validate_loaded_critic,
)
from oasis_rc_v2.checkpoint import (
    CHECKPOINT_SCHEMA,
    EXPERIMENT_ID,
    IMPLEMENTATION_VERSION,
    METHOD_VERSION,
    validate_critic_checkpoint,
)
from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.losses import segmentation_loss, oasis_rc_student_loss_v2


def flat(loss, params, retain=False):
    grads = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).reshape(-1)
            for p, g in zip(params, grads)
        ]
    )


def fixture():
    torch.manual_seed(123)
    student = MultiScaleLightweightSegmenter(width=4)
    x = torch.randn(2, 3, 32, 32)
    y = torch.zeros(2, 1, 32, 32)
    y[:, :, 14:18, 4:28] = 1
    return student, x, y


def test_s0_equals_s2_when_lambda_aosk_zero():
    student, x, y = fixture()
    params = [p for p in student.parameters() if p.requires_grad]
    logits = student(x)
    seg = segmentation_loss(logits, y)
    aosk = oriented_consistency_loss(logits, x, y)
    assert torch.equal(flat(seg, params, True), flat(seg + 0 * aosk, params))


def test_s1_equals_s3_when_lambda_aosk_zero():
    student, x, y = fixture()
    params = [p for p in student.parameters() if p.requires_grad]
    critic = OASISRCv2Critic(width=4).eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    logits = student(x)
    seg = segmentation_loss(logits, y)
    pred = logits.sigmoid()
    wrong, _ = make_corrupted_mask(
        y, generator=make_generator(torch.device("cpu"), 17)
    )
    with torch.no_grad():
        gt = critic(x, y)
        corrupt = critic(x, wrong)
    rc, extras = oasis_rc_student_loss_v2(critic(x, pred), gt, corrupt, pred, y)
    aosk = oriented_consistency_loss(logits, x, y)
    assert torch.equal(
        flat(seg + 0.001 * rc, params, True),
        flat(seg + 0.001 * rc + 0 * aosk, params),
    )
    assert all(
        key in extras
        for key in (
            "e_pred",
            "e_gt",
            "e_corrupted",
            "delta_pred_gt",
            "delta_pred_corrupted",
        )
    )


def test_rc_corruption_rng_does_not_change_augmentation_sequence():
    x = torch.linspace(-1, 1, 2 * 3 * 16 * 16).reshape(2, 3, 16, 16)
    y = torch.zeros(2, 1, 16, 16)
    y[:, :, 6:10, 3:13] = 1
    control = make_generator(torch.device("cpu"), 777)
    connected = make_generator(torch.device("cpu"), 777)
    corruption = make_generator(torch.device("cpu"), 999)
    c1 = augment(x.clone(), y.clone(), control)
    c2 = augment(x.clone(), y.clone(), control)
    r1 = augment(x.clone(), y.clone(), connected)
    make_corrupted_mask(y, generator=corruption)
    r2 = augment(x.clone(), y.clone(), connected)
    assert torch.equal(c1[0], r1[0]) and torch.equal(c1[1], r1[1])
    assert torch.equal(c2[0], r2[0]) and torch.equal(c2[1], r2[1])


def test_noop_target_is_pair_valid():
    mask = torch.zeros(2, 1, 8, 8)
    _, mismatch, pair_valid = build_targets(mask, torch.zeros_like(mask))
    assert float(mismatch.sum()) == 0
    assert torch.equal(pair_valid, torch.ones_like(pair_valid))


def test_two_step_zero_rc_equivalence():
    base, x, y = fixture()
    control = copy.deepcopy(base)
    connected = copy.deepcopy(base)
    critic = OASISRCv2Critic(width=4).eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    opt_control = torch.optim.AdamW(control.parameters(), lr=1e-4)
    opt_connected = torch.optim.AdamW(connected.parameters(), lr=1e-4)
    gen_control = make_generator(torch.device("cpu"), 101)
    gen_connected = make_generator(torch.device("cpu"), 101)
    gen_corrupt = make_generator(torch.device("cpu"), 202)
    for _ in range(2):
        xc, yc = augment(x.clone(), y.clone(), gen_control)
        xr, yr = augment(x.clone(), y.clone(), gen_connected)
        loss_control = segmentation_loss(control(xc), yc)
        opt_control.zero_grad()
        loss_control.backward()
        opt_control.step()

        logits = connected(xr)
        seg = segmentation_loss(logits, yr)
        pred = logits.sigmoid()
        wrong, _ = make_corrupted_mask(yr, generator=gen_corrupt)
        with torch.no_grad():
            gt = critic(xr, yr)
            corrupt = critic(xr, wrong)
        rc, _ = oasis_rc_student_loss_v2(critic(xr, pred), gt, corrupt, pred, yr)
        loss_connected = seg + 0 * rc
        opt_connected.zero_grad()
        loss_connected.backward()
        opt_connected.step()

    for (name_a, tensor_a), (name_b, tensor_b) in zip(
        control.state_dict().items(), connected.state_dict().items()
    ):
        assert name_a == name_b
        assert torch.equal(tensor_a, tensor_b), name_a


def test_student_init_seed_mismatch_is_rejected(tmp_path):
    student = MultiScaleLightweightSegmenter(width=4)
    setattr(student, "_oasis_width", 4)
    path = tmp_path / "init.pt"
    torch.save(
        {
            "student": student.state_dict(),
            "student_kind": "multiscale",
            "student_width": 4,
            "seed": 1337,
        },
        path,
    )
    with pytest.raises(ValueError, match="seed mismatch"):
        load_student_init(student, path, 2027)


def _qualified_saved_critic(manifest, full_certificate, dataset_sha, cfg):
    training_hparams = {
        "crack_dice_weight": 1.0,
        "mismatch_weight": 1.0,
        "pair_weight": 0.25,
        "rgb_mask_weight": 1.0,
        "normal_critic_weight": 1.0,
        "normal_fraction": 0.25,
        "rgb_shuffle_pair_only": True,
        "mask_flip_training": False,
        "mask_variant_contract": "operator-preserved-v1",
        "energy_head_contract": "dedicated-scalar-lower-is-better-v1",
        "method_spec": "METHOD_SPEC_V2_1.md",
    }
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "method_version": METHOD_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "energy_head_contract": "dedicated-scalar-lower-is-better-v1",
        "critic": {},
        "width": 8,
        "seed": 1337,
        "full_gate0_certificate_sha256": sha256_file(full_certificate),
        "config": cfg,
        "manifest_file_sha256": sha256_file(manifest),
        "dataset_content_sha256": dataset_sha,
        "normal_fraction": 0.25,
        "normal_critic_weight": 1.0,
        "training_hparams": training_hparams,
        "qualification_v21": {"pass": True, "failures": []},
    }


def test_v21_critic_provenance_and_schema_fail_closed(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("canonical-bytes\n")
    full_certificate = tmp_path / "full_gate0.json"
    full_certificate.write_text('{"status":"PASS","scope":"full_benchmark"}')
    dataset_sha = "d" * 64
    cfg = {"seed": 1337, "image_size": 256}
    saved = _qualified_saved_critic(manifest, full_certificate, dataset_sha, cfg)
    expected = {
        key: saved["training_hparams"][key]
        for key in (
            "energy_head_contract",
            "mask_variant_contract",
            "rgb_shuffle_pair_only",
            "mask_flip_training",
            "method_spec",
        )
    }
    validate_critic_checkpoint(
        saved,
        manifest,
        cfg,
        0.25,
        1.0,
        dataset_content_sha256_value=dataset_sha,
        expected_hparams=expected,
        full_gate0_certificate=full_certificate,
    )

    bad = dict(saved)
    bad.pop("checkpoint_schema")
    with pytest.raises(ValueError, match="legacy checkpoint rejected"):
        validate_critic_checkpoint(
            bad,
            manifest,
            cfg,
            0.25,
            1.0,
            expected_hparams=expected,
            full_gate0_certificate=full_certificate,
        )

    bad_impl = dict(saved)
    bad_impl["implementation_version"] = "legacy"
    with pytest.raises(ValueError, match="implementation_version"):
        validate_critic_checkpoint(
            bad_impl,
            manifest,
            cfg,
            0.25,
            1.0,
            expected_hparams=expected,
            full_gate0_certificate=full_certificate,
        )

    bad_data = dict(saved)
    bad_data["dataset_content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="dataset-content"):
        validate_critic_checkpoint(
            bad_data,
            manifest,
            cfg,
            0.25,
            1.0,
            dataset_content_sha256_value=dataset_sha,
            expected_hparams=expected,
            full_gate0_certificate=full_certificate,
        )

    bad_contract = copy.deepcopy(saved)
    bad_contract["training_hparams"].pop("mask_flip_training")
    with pytest.raises(ValueError, match="mask_flip_training"):
        validate_critic_checkpoint(
            bad_contract,
            manifest,
            cfg,
            0.25,
            1.0,
            expected_hparams=expected,
            full_gate0_certificate=full_certificate,
        )


def test_legacy_consumer_fails_closed_on_v21_dev2_critic(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("canonical-bytes\n")
    full_certificate = tmp_path / "full_gate0.json"
    full_certificate.write_text('{"status":"PASS","scope":"full_benchmark"}')
    dataset_sha = "d" * 64
    cfg = {"seed": 1337, "image_size": 256}
    saved = _qualified_saved_critic(manifest, full_certificate, dataset_sha, cfg)
    args = SimpleNamespace(
        manifest=str(manifest),
        normal_fraction=0.25,
        normal_critic_weight=1.0,
        crack_dice_weight=1.0,
        mismatch_weight=1.0,
        pair_weight=0.25,
        rgb_mask_weight=1.0,
        _dataset_content_sha256=dataset_sha,
        full_gate0_certificate=str(full_certificate),
    )
    with pytest.raises(ValueError, match="full v2.1 contract"):
        validate_loaded_critic(saved, args, cfg)


def test_sha256_exact(tmp_path):
    path = tmp_path / "x"
    path.write_bytes(b"a")
    assert sha256_file(path) == hashlib.sha256(b"a").hexdigest()
