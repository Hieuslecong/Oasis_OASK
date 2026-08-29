import torch

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
from oasis_cycle_aosk.evaluate_a2s import _verify_manifest_provenance
from oasis_cycle_aosk.train_a2s import _assert_dev_split, _sha256_json, build_parser


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
        "test",
        "final",
        "holdout",
        "external_test",
        "final_test",
        "final_external",
        "test_2026",
        "external-final",
        "holdout_v2",
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


def test_stage1_defaults_match_oasis_reference_learning_rates():
    args = build_parser().parse_args(["--manifest", "m", "--out", "o"])
    assert args.lr_d == 4e-4
    assert args.lr_g == 1e-4
    assert args.lambda_labelmix == 10.0
    assert args.allow_dirty is False
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
    ck = {
        "method": "OASIS-A2S-v0.1",
        "arm": "A2",
        "segmenter": s.state_dict(),
        "generator_in_checkpoint": False,
    }
    assert not ({"generator", "discriminator", "critic", "aosk"} & set(ck))
