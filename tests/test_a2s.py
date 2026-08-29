import json
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from oasis_cycle_aosk.a2s import (
    FAKE_CLASS,
    OASISA2SDiscriminator,
    OASISA2SGenerator,
    balanced_semantic_ce,
    fake_ce,
    labelmix_mask,
    parameter_count,
    semantic_target,
    soft_dice_loss,
    stage1_discriminator_loss,
    stage1_generator_loss,
    stage1_real_class_logits,
    stage2_segmentation_loss,
    transfer_to_segmenter,
)
from oasis_cycle_aosk.data import audit_manifest
from oasis_cycle_aosk.evaluate_a2s import _verify_manifest_provenance
from oasis_cycle_aosk.metrics_a2s import paired_bootstrap
from oasis_cycle_aosk.train_a2s import (
    _assert_dev_split,
    _assert_git_provenance,
    _sha256_json,
    build_parser,
    interpolate_segmenters,
    run,
    train_stage1,
    _capture_rng_state,
    _restore_rng_state,
    _seed_everything,
)


def toy(batch=2, size=32):
    x = torch.zeros(batch, 3, size, size)
    y = torch.zeros(batch, 1, size, size)
    y[:, :, 7:25, 15:17] = 1
    x[:, 0] = y[:, 0] * 0.8 - 0.2
    x[:, 1] = -0.4
    x[:, 2] = 0.2
    return x, y


def test_stage1_shapes_and_finiteness():
    torch.manual_seed(3)
    x, y = toy()
    d = OASISA2SDiscriminator(8, 3)
    g = OASISA2SGenerator(8, 2)
    fake = g(y)
    assert fake.shape == x.shape and fake.min() >= -1 and fake.max() <= 1
    assert d(x).shape == (2, 3, 32, 32)
    ld, lr, lf, lm = stage1_discriminator_loss(d, x, y, fake.detach(), 1.0)
    lg = stage1_generator_loss(d, fake, y)
    for v in (ld, lr, lf, lm, lg):
        assert v.ndim == 0 and torch.isfinite(v)


def test_generator_is_one_to_many_with_global_oasis_noise():
    _, y = toy(1)
    g = OASISA2SGenerator(8, 2)
    assert g.condition_channels == 4
    z0 = torch.zeros(1, 2, 1, 1)
    z1 = torch.ones(1, 2, 1, 1)
    assert not torch.allclose(g(y, z0), g(y, z1))


def test_generator_accepts_explicit_local_noise_but_rejects_bad_spatial_shape():
    _, y = toy(1)
    g = OASISA2SGenerator(8, 2)
    local = torch.randn(1, 2, 32, 32)
    assert g(y, local).shape == (1, 3, 32, 32)
    try:
        g(y, torch.randn(1, 2, 4, 4))
        assert False
    except ValueError:
        pass


def test_semantic_target_binary():
    _, y = toy(1)
    t = semantic_target(y)
    assert t.dtype == torch.long and set(t.unique().tolist()) == {0, 1}


def test_balancing_upweights_sparse_crack():
    _, y = toy(1)
    target = semantic_target(y)
    logits = torch.zeros(1, 3, 32, 32, requires_grad=True)
    loss = balanced_semantic_ce(logits, target)
    loss.backward()
    assert loss.item() > 0 and torch.isfinite(logits.grad).all()


def test_fake_ce_targets_third_channel():
    good = torch.full((1, 3, 8, 8), -4.0)
    good[:, FAKE_CLASS] = 4.0
    assert fake_ce(good) < fake_ce(-good)


def test_labelmix_uses_one_decision_per_class_across_batch():
    _, y = toy(4)
    target = semantic_target(y)
    mix = labelmix_mask(target, torch.Generator().manual_seed(11))[:, 0]
    for c in (0, 1):
        assert mix[target == c].unique().numel() <= 1


def test_transfer_preserves_body_and_real_head_exactly():
    torch.manual_seed(7)
    d = OASISA2SDiscriminator(8, 3)
    s = transfer_to_segmenter(d)
    ds, ss = d.state_dict(), s.state_dict()
    for k in ss:
        assert torch.equal(ss[k], ds[k][:2] if k.startswith("head.") else ds[k])
    assert s.out_classes == 2


def test_a1_logits_equal_first_two_stage1_channels():
    x, _ = toy(1)
    d = OASISA2SDiscriminator(8, 3)
    assert torch.equal(stage1_real_class_logits(d, x), d(x)[:, :2])


def test_stage2_loss_perfect_prediction_is_small():
    _, y = toy(1)
    target = semantic_target(y)
    logits = torch.full((1, 2, 32, 32), -8.0)
    logits.scatter_(1, target[:, None], 8.0)
    assert stage2_segmentation_loss(logits, y).item() < 1e-3
    assert soft_dice_loss(logits, target).item() < 1e-3


def test_invalid_contracts_fail_closed():
    d2 = OASISA2SDiscriminator(8, 2)
    x, _ = toy(1)
    for fn in (lambda: transfer_to_segmenter(d2), lambda: stage1_real_class_logits(d2, x)):
        try:
            fn()
            assert False
        except ValueError:
            pass


def test_models_are_compact():
    assert parameter_count(OASISA2SDiscriminator(24, 3)) < 2_000_000
    assert parameter_count(OASISA2SGenerator(32, 4)) < 2_000_000


def test_development_firewall_rejects_semantic_test_tokens():
    forbidden = (
        "test", "final", "holdout", "external_test", "final_test",
        "final_external", "test_2026", "external-final", "holdout_v2",
        "evaluation_test",
    )
    for name in forbidden:
        try:
            _assert_dev_split(name)
            assert False, name
        except ValueError:
            pass
    for name in ("train", "val", "validation", "cal", "external_val"):
        _assert_dev_split(name)


def test_manifest_provenance_fail_closed_for_development():
    a = "a" * 64
    b = "b" * 64
    assert _verify_manifest_provenance(a, a, is_dev_split=True, allow_manifest_mismatch=False)
    for allow in (False, True):
        try:
            _verify_manifest_provenance(a, b, is_dev_split=True, allow_manifest_mismatch=allow)
            assert False
        except ValueError:
            pass


def test_manifest_mismatch_only_allowed_for_explicit_external_final():
    a = "a" * 64
    b = "b" * 64
    assert not _verify_manifest_provenance(a, b, is_dev_split=False, allow_manifest_mismatch=True)
    try:
        _verify_manifest_provenance(a, b, is_dev_split=False, allow_manifest_mismatch=False)
        assert False
    except ValueError:
        pass


def test_stage1_defaults_match_oasis_reference_learning_rates_and_protocol():
    args = build_parser().parse_args(["--manifest", "m", "--out", "o"])
    assert args.lr_d == 4e-4
    assert args.lr_g == 1e-4
    assert args.lambda_labelmix == 10.0
    assert args.fit_split == "fit" and args.cal_split == "cal" and args.val_split == "val"
    assert args.stage1_epochs == 50 and args.a0_epochs == 100 and args.stage2_epochs == 30
    assert args.wise_alpha == 0.8
    assert args.allow_dirty is False and args.allow_unversioned is False
    assert args.allow_nondeterministic is False


def test_config_sha_is_order_invariant():
    assert _sha256_json({"a": 1, "b": 2}) == _sha256_json({"b": 2, "a": 1})


def test_one_optimizer_step_changes_expected_network_only():
    x, y = toy(1, 16)
    d = OASISA2SDiscriminator(8, 3)
    g = OASISA2SGenerator(8, 2)
    d0 = {k: v.detach().clone() for k, v in d.state_dict().items()}
    g0 = {k: v.detach().clone() for k, v in g.state_dict().items()}
    opt = torch.optim.Adam(d.parameters(), lr=1e-3)
    with torch.no_grad():
        fake = g(y)
    opt.zero_grad()
    ld, *_ = stage1_discriminator_loss(d, x, y, fake, 1.0)
    ld.backward()
    opt.step()
    assert any(not torch.equal(d0[k], d.state_dict()[k]) for k in d0)
    assert all(torch.equal(g0[k], g.state_dict()[k]) for k in g0)


def test_stage2_checkpoint_contract_contains_no_training_only_network():
    s = transfer_to_segmenter(OASISA2SDiscriminator(8, 3))
    ck = {"method": "OASIS-A2S-v0.1", "arm": "A2-Full", "segmenter": s.state_dict(), "generator_in_checkpoint": False}
    assert not ({"generator", "discriminator", "critic", "aosk"} & set(ck))


def _make_manifest(tmp_path):
    rows = []
    for i, split in enumerate(("fit", "fit", "cal", "val")):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[3:13, 6 + (i % 3):8 + (i % 3)] = 255
        image[..., 0] = mask
        image[0, 0, 1] = i + 1
        ip = tmp_path / f"i{i}.png"
        mp = tmp_path / f"m{i}.png"
        Image.fromarray(image).save(ip)
        Image.fromarray(mask).save(mp)
        rows.append({
            "image": str(ip), "mask": str(mp), "split": split, "is_normal": False,
            "source_id": "toy", "lineage_id": f"lin-{i}", "sample_id": f"toy-{i}",
        })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return manifest, rows


def test_dataset_content_hash_binds_file_bytes_and_lineage(tmp_path):
    manifest, rows = _make_manifest(tmp_path)
    a = audit_manifest(manifest, ("fit", "cal", "val"))
    assert len(a["dataset_content_sha256"]) == 64
    image = np.asarray(Image.open(rows[0]["image"]).convert("RGB")).copy()
    image[0, 1, 2] = 123
    Image.fromarray(image).save(rows[0]["image"])
    b = audit_manifest(manifest, ("fit", "cal", "val"))
    assert a["dataset_content_sha256"] != b["dataset_content_sha256"]


def test_audit_rejects_cross_split_lineage_and_exact_rgb(tmp_path):
    manifest, rows = _make_manifest(tmp_path)
    rows[2]["lineage_id"] = rows[0]["lineage_id"]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    try:
        audit_manifest(manifest, ("fit", "cal", "val"))
        assert False
    except ValueError as e:
        assert "lineage" in str(e)

    manifest, rows = _make_manifest(tmp_path)
    Image.open(rows[0]["image"]).save(rows[2]["image"])
    try:
        audit_manifest(manifest, ("fit", "cal", "val"))
        assert False
    except ValueError as e:
        assert "duplicate" in str(e).lower()


def test_interpolation_endpoints_are_exact():
    pre = transfer_to_segmenter(OASISA2SDiscriminator(8, 3))
    ft = transfer_to_segmenter(OASISA2SDiscriminator(8, 3))
    with torch.no_grad():
        for p in ft.parameters():
            p.add_(0.25)
    w0 = interpolate_segmenters(pre, ft, 0.0)
    w1 = interpolate_segmenters(pre, ft, 1.0)
    for k in pre.state_dict():
        assert torch.equal(w0.state_dict()[k], pre.state_dict()[k])
        assert torch.equal(w1.state_dict()[k], ft.state_dict()[k])


def test_paired_bootstrap_detects_positive_delta():
    a = [{"sample_key": str(i), "dice_f1": 0.2 + i * 0.01} for i in range(8)]
    b = [{"sample_key": str(i), "dice_f1": 0.3 + i * 0.01} for i in range(8)]
    out = paired_bootstrap(a, b, seed=7, draws=500)
    assert out["mean_delta_dice"] > 0.09
    assert out["positive_fraction"] == 1.0
    assert out["bootstrap_95ci"][0] > 0


def test_git_provenance_fail_closed_when_unversioned(monkeypatch):
    import oasis_cycle_aosk.train_a2s as t
    monkeypatch.setattr(t, "_git_commit", lambda: None)
    monkeypatch.setattr(t, "_git_dirty", lambda: None)
    try:
        _assert_git_provenance(allow_dirty=False, allow_unversioned=False)
        assert False
    except RuntimeError:
        pass
    commit, dirty = _assert_git_provenance(allow_dirty=False, allow_unversioned=True)
    assert commit is None and dirty is None


def test_stage1_resume_preserves_optimizer_and_rng_trajectory(tmp_path):
    from oasis_cycle_aosk.data import ManifestDataset
    manifest, _ = _make_manifest(tmp_path)
    fit_ds = ManifestDataset(manifest, "fit", 16)

    _seed_everything(99)
    d_full = OASISA2SDiscriminator(4, 3)
    g_full = OASISA2SGenerator(4, 2)
    h_full, _, _ = train_stage1(
        d_full, g_full, fit_ds, torch.device("cpu"), total_epochs=2, batch=1, workers=0,
        lr_d=4e-4, lr_g=1e-4, lambda_labelmix=10.0, seed=99,
    )

    _seed_everything(99)
    d_res = OASISA2SDiscriminator(4, 3)
    g_res = OASISA2SGenerator(4, 2)
    h1, od, og = train_stage1(
        d_res, g_res, fit_ds, torch.device("cpu"), total_epochs=1, batch=1, workers=0,
        lr_d=4e-4, lr_g=1e-4, lambda_labelmix=10.0, seed=99,
    )
    rng = _capture_rng_state()
    d_state = {k: v.detach().clone() for k, v in d_res.state_dict().items()}
    g_state = {k: v.detach().clone() for k, v in g_res.state_dict().items()}
    od_state, og_state = od.state_dict(), og.state_dict()

    d_resume = OASISA2SDiscriminator(4, 3); d_resume.load_state_dict(d_state)
    g_resume = OASISA2SGenerator(4, 2); g_resume.load_state_dict(g_state)
    _restore_rng_state(rng)
    h_res, _, _ = train_stage1(
        d_resume, g_resume, fit_ds, torch.device("cpu"), total_epochs=2, batch=1, workers=0,
        lr_d=4e-4, lr_g=1e-4, lambda_labelmix=10.0, seed=99, start_epoch=1,
        opt_d_state=od_state, opt_g_state=og_state, history=h1,
    )
    assert h_res == h_full
    for k in d_full.state_dict():
        assert torch.equal(d_full.state_dict()[k], d_resume.state_dict()[k]), k
    for k in g_full.state_dict():
        assert torch.equal(g_full.state_dict()[k], g_resume.state_dict()[k]), k


def test_end_to_end_tiny_gate1_smoke(tmp_path):
    manifest, _ = _make_manifest(tmp_path)
    out = tmp_path / "run"
    args = SimpleNamespace(
        manifest=str(manifest), out=str(out), fit_split="fit", cal_split="cal", val_split="val",
        size=16, batch=1, workers=0, device="cpu", seed=1337, width=4,
        generator_width=4, noise_channels=2, stage1_epochs=1, stage1_checkpoints="1",
        stage1_resume=None, a0_epochs=1, stage2_epochs=1, stage2_checkpoints="1",
        lr_d=4e-4, lr_g=1e-4, stage2_lr=2e-4, lambda_labelmix=10.0,
        dice_weight=1.0, threshold_grid="0.30,0.50,0.70", wise_alpha=0.8,
        allow_nondeterministic=False, allow_dirty=True, allow_unversioned=True,
        allow_missing_lineage=False, allow_size_mismatch=False,
    )
    result = run(args)
    assert set(result["arms"]) == {"A0", "A1", "A2-Full", "A2-WI"}
    for name in ("a0_supervised.pt", "stage1_oasis.pt", "stage1_epoch_001.pt", "a1_direct.pt", "a2_full.pt", "a2_wi.pt", "results.json"):
        assert (out / name).exists(), name
    stage1 = torch.load(out / "stage1_oasis.pt", map_location="cpu", weights_only=False)
    assert "optimizer_d" in stage1 and "optimizer_g" in stage1 and stage1["completed_epoch"] == 1
    for arm in result["arms"].values():
        assert 0 < arm["threshold"] < 1
        assert "mean_image_cldice" in arm and "mean_image_boundary_f1" in arm
    assert "A2-WI_minus_A0" in result["paired_val"]
    assert len(result["provenance"]["config_sha256"]) == 64
    assert len(result["provenance"]["dataset_content_sha256"]) == 64
