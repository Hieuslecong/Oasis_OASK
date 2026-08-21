import copy
import json

import pytest

torch = pytest.importorskip("torch")

from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.energy_qualification import summarize_energy_trajectory
from oasis_rc_v2.losses import (
    continuous_relation_path_loss,
    critic_endpoint_energy_loss,
    oasis_rc_critic_loss,
    oasis_rc_student_loss_v2,
    segmentation_loss,
)
from oasis_rc_v2.corruptions import build_targets
from oasis_cycle_aosk.aosk import oriented_consistency_loss
from oasis_cycle_aosk.models import LightweightSegmenter


def _batch():
    g = torch.Generator().manual_seed(123)
    y = torch.zeros((4, 1, 32, 32))
    y[0, 0, 7:25, 15:17] = 1
    y[1, 0, 15:17, 6:26] = 1
    y[2, 0, 7:25, 10:12] = 1
    y[2, 0, 15:17, 10:24] = 1
    y[3, 0, 8:24, 20:22] = 1
    x = torch.rand((4, 3, 32, 32), generator=g) * 0.10 - 0.05
    x = (x + y.repeat(1, 3, 1, 1) * 0.90).clamp(-1, 1)
    wrong = torch.zeros_like(y)
    wrong[..., 3:] = y[..., :-3]
    changed = (wrong - y).abs().flatten(1).sum(1) > 0
    assert bool(changed.all())
    return x, y, wrong


def _critic_term(critic, x, mask, invalid):
    semantic, mismatch, pair = build_targets(mask, invalid)
    return oasis_rc_critic_loss(critic(x, mask), semantic, mismatch, pair)[0]


def _train_critic():
    x, y, wrong = _batch()
    critic = OASISRCv2Critic(width=4)
    opt = torch.optim.AdamW(critic.parameters(), lr=3e-3, weight_decay=0.0)
    history = []
    # Keep the release gate fixed; give the tiny CPU critic enough optimization
    # steps to prove capacity to learn the ordered soft-mask landscape.
    for _ in range(96):
        clean = _critic_term(critic, x, y, torch.zeros_like(y))
        corrupt = _critic_term(critic, x, wrong, (wrong-y).abs())
        semantic, mismatch, pair = build_targets(y, torch.zeros_like(y))
        rgb_pair, _ = oasis_rc_critic_loss(
            critic(x.flip(-1), y), semantic, mismatch, torch.zeros_like(pair)
        )
        endpoint, endpoint_terms = critic_endpoint_energy_loss(
            critic, x, y, wrong, margin=0.05, anchor_weight=0.25
        )
        path, path_terms = continuous_relation_path_loss(
            critic, x, y, wrong, margin=0.02
        )
        loss = 0.5*(clean+corrupt) + rgb_pair + endpoint + path
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append({
            "loss": float(loss.detach()),
            "endpoint": float(endpoint.detach()),
            "path": float(path.detach()),
            "endpoint_order_fraction": float(endpoint_terms["endpoint_order_fraction"]),
            "path_order_fraction": float(path_terms["path_order_fraction"]),
        })
    for row in history:
        assert all(torch.isfinite(torch.tensor(v)) for v in row.values())
    metrics = summarize_energy_trajectory(
        critic, x, y, wrong,
        levels=(0.0,0.25,0.5,0.75,1.0), margin=0.02,
    )
    print("V21_ENERGY_DEBUG=" + json.dumps(metrics, sort_keys=True), flush=True)
    assert metrics["energy_finite"] is True
    assert metrics["positive_energy_gap_fraction"] >= 0.90
    assert metrics["continuous_path_order_fraction"] >= 0.80
    assert metrics["mean_energy_gap"] > 0
    assert metrics["median_energy_gap"] > 0
    return critic, history, metrics


def _one_student_step(mode, critic, initial_state):
    x, y, wrong = _batch()
    student = LightweightSegmenter(width=4)
    student.load_state_dict(copy.deepcopy(initial_state))
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.0)
    critic.eval()
    critic.zero_grad(set_to_none=True)
    for p in critic.parameters():
        p.requires_grad_(False)
    logits = student(x)
    seg = segmentation_loss(logits, y)
    total = seg
    aux = logits.new_zeros(())
    if mode in {"S1", "S3"}:
        pred = logits.sigmoid()
        with torch.no_grad():
            gt_out = critic(x, y)
            corrupt_out = critic(x, wrong)
        aux, extras = oasis_rc_student_loss_v2(
            critic(x, pred), gt_out, corrupt_out, pred, y, margin=0.1
        )
        assert torch.isfinite(aux)
        assert all(torch.isfinite(v).all() for v in extras.values() if torch.is_tensor(v))
        total = total + 0.001 * aux
    if mode in {"S2", "S3"}:
        aosk = oriented_consistency_loss(logits, x, y)
        assert torch.isfinite(aosk)
        total = total + 0.01 * aosk
    opt.zero_grad(set_to_none=True)
    total.backward()
    grad_norm = torch.sqrt(sum((p.grad.detach()**2).sum() for p in student.parameters() if p.grad is not None))
    assert torch.isfinite(grad_norm) and float(grad_norm) > 0
    assert all(p.grad is None for p in critic.parameters())
    before = {k: v.detach().clone() for k, v in student.state_dict().items()}
    opt.step()
    assert any(not torch.equal(before[k], v) for k, v in student.state_dict().items())
    return {
        "mode": mode,
        "loss_total": float(total.detach()),
        "loss_seg": float(seg.detach()),
        "loss_aux": float(aux.detach()),
        "grad_norm": float(grad_norm),
    }


def test_v21_cpu_smoke_critic_and_four_arms():
    torch.manual_seed(17)
    critic, critic_history, energy = _train_critic()
    base = LightweightSegmenter(width=4)
    initial = copy.deepcopy(base.state_dict())
    results = [_one_student_step(mode, critic, initial) for mode in ("S0", "S1", "S2", "S3")]
    assert [r["mode"] for r in results] == ["S0", "S1", "S2", "S3"]
    assert all(r["loss_total"] > 0 for r in results)
    summary = {
        "critic_final": critic_history[-1],
        "energy_validation": energy,
        "arms": results,
        "canonical_test_opened": False,
    }
    print("V21_SMOKE_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
