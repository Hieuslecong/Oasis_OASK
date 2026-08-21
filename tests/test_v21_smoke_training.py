import copy
import json

import pytest

torch = pytest.importorskip("torch")

from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.losses import (
    continuous_relation_path_loss,
    oasis_rc_critic_loss,
    oasis_rc_student_loss_v2,
    segmentation_loss,
)
from oasis_rc_v2.corruptions import build_targets
from oasis_cycle_aosk.aosk import oriented_consistency_loss
from oasis_cycle_aosk.models import LightweightSegmenter


def _batch():
    g = torch.Generator().manual_seed(123)
    x = torch.rand((4, 3, 32, 32), generator=g) * 2 - 1
    y = torch.zeros((4, 1, 32, 32))
    y[0, 0, 7:25, 15:17] = 1
    y[1, 0, 15:17, 6:26] = 1
    y[2, 0, 7:25, 10:12] = 1
    y[2, 0, 15:17, 10:24] = 1
    y[3, 0, 8:24, 20:22] = 1
    wrong = y.flip(-1)
    return x, y, wrong


def _train_critic():
    x, y, wrong = _batch()
    critic = OASISRCv2Critic(width=4)
    opt = torch.optim.AdamW(critic.parameters(), lr=2e-3, weight_decay=0.0)
    invalid = (wrong - y).abs()
    semantic, mismatch, pair = build_targets(wrong, invalid)
    history = []
    for _ in range(4):
        out = critic(x, wrong)
        cls, _ = oasis_rc_critic_loss(out, semantic, mismatch, pair)
        path, terms = continuous_relation_path_loss(
            critic, x, y, wrong, pair_weight=0.25, margin=0.01
        )
        loss = cls + 0.25 * path
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append((float(loss.detach()), float(path.detach()), float(terms["path_order_fraction"])))
    assert all(torch.isfinite(torch.tensor(v)).all() for row in history for v in row)
    return critic, history


def _one_student_step(mode, critic, initial_state):
    x, y, wrong = _batch()
    student = LightweightSegmenter(width=4)
    student.load_state_dict(copy.deepcopy(initial_state))
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.0)
    critic.eval()
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
            critic(x, pred), gt_out, corrupt_out, pred, y,
            margin=0.1, pair_weight=0.25,
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
    changed = any(not torch.equal(before[k], v) for k, v in student.state_dict().items())
    assert changed
    return {
        "mode": mode,
        "loss_total": float(total.detach()),
        "loss_seg": float(seg.detach()),
        "loss_aux": float(aux.detach()),
        "grad_norm": float(grad_norm),
    }


def test_v21_cpu_smoke_critic_and_four_arms(capsys):
    torch.manual_seed(17)
    critic, critic_history = _train_critic()
    base = LightweightSegmenter(width=4)
    initial = copy.deepcopy(base.state_dict())
    results = [_one_student_step(mode, critic, initial) for mode in ("S0", "S1", "S2", "S3")]
    assert [r["mode"] for r in results] == ["S0", "S1", "S2", "S3"]
    assert all(r["loss_total"] > 0 for r in results)
    summary = {
        "critic_final_loss": critic_history[-1][0],
        "critic_final_path_loss": critic_history[-1][1],
        "critic_final_path_order_fraction": critic_history[-1][2],
        "arms": results,
        "canonical_test_opened": False,
    }
    print("V21_SMOKE_RESULT=" + json.dumps(summary, sort_keys=True))
    captured = capsys.readouterr().out
    assert "V21_SMOKE_RESULT=" in captured
