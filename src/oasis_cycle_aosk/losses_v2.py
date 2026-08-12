import torch
import torch.nn.functional as F

def dice_loss(logits, target, eps=1e-6):
    p = logits.sigmoid(); inter = (p * target).sum((1,2,3)); den = p.sum((1,2,3)) + target.sum((1,2,3))
    return (1 - (2 * inter + eps) / (den + eps)).mean()

def segmentation_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)

def oasis_hard_negative_loss(student_logits, target, critic_logits):
    """Penalize student crack predictions only where GT is background and OASIS
    assigns high background confidence. Critic confidence is detached so this
    term cannot train the critic through the student update.
    """
    p_student = student_logits.sigmoid()
    p_background = critic_logits.softmax(dim=1)[:, 0:1].detach()
    background = 1.0 - target
    return (background * p_student * p_background).mean()

def conditional_oasis_loss(critic_logits, target):
    """Make a student mask/image pair semantically valid according to a frozen critic."""
    return F.cross_entropy(critic_logits, target[:, 0].long(), weight=critic_logits.new_tensor([1.0, 8.0, 1.0]))


def oasis_rc_critic_loss(out, semantic_target, mismatch_target, pair_valid,
                         class_weight=None, pair_weight=0.25):
    """Balanced loss for OASIS-RC critic; targets are detached labels."""
    if class_weight is None:
        class_weight = out["semantic"].new_tensor([1.0, 20.0, 12.0])
    sem = F.cross_entropy(out["semantic"], semantic_target.long(), weight=class_weight)
    # Preserve the valid crack/background semantics separately from the
    # invalid class.  Otherwise the high-volume invalid negatives can make the
    # 3-way head collapse to background/invalid while still looking accurate.
    valid = pair_valid.view(-1, 1, 1, 1)
    crack_target = (semantic_target == 1).float().unsqueeze(1)
    crack_logit = out.get("crack", out["semantic"][:, 1:2] - out["semantic"][:, 0:1])
    valid_count = valid.sum().clamp_min(1.0)
    valid_crack = (crack_target * valid).sum().clamp_min(1.0)
    valid_bg = ((1.0 - crack_target) * valid).sum().clamp_min(1.0)
    crack_pw = (valid_bg / valid_crack).clamp(5.0, 50.0).detach()
    crack_aux = (F.binary_cross_entropy_with_logits(crack_logit, crack_target, pos_weight=crack_pw, reduction="none") * valid).sum() / (valid_count * crack_target.shape[-2] * crack_target.shape[-1])
    # Mismatch pixels are sparse. BCEWithLogits with a conservative positive
    # weight avoids learning an all-zero mismatch map.
    pos = mismatch_target.sum().clamp_min(1.0)
    neg = mismatch_target.numel() - pos
    pw = (neg / pos).clamp(1.0, 20.0).detach()
    mismatch = F.binary_cross_entropy_with_logits(out["mismatch"], mismatch_target, pos_weight=pw)
    valid_crack_pixels = valid * crack_target
    valid_crack_count = valid_crack_pixels.sum().clamp_min(1.0)
    clean_mismatch = (F.binary_cross_entropy_with_logits(
        out["mismatch"], torch.zeros_like(mismatch_target), reduction="none"
    ) * valid_crack_pixels).sum() / valid_crack_count
    pair = F.binary_cross_entropy_with_logits(out["pair"], pair_valid, pos_weight=out["pair"].new_tensor(1.0))
    return sem + 2.0 * crack_aux + mismatch + 2.0 * clean_mismatch + pair_weight * pair


def oasis_rc_student_loss(pred_out, gt_out, student_mask, target, margin=0.10):
    """Contrastive OASIS loss for the student, with critic parameters frozen.

    The ranking term asks the critic to score the prediction as no worse than a
    deliberately invalid pair, while the FP term acts only on GT background.
    No gradient is allowed through the GT target or through the critic weights.
    """
    q_pred = pred_out["mismatch"].sigmoid()
    q_gt = gt_out["mismatch"].sigmoid().detach()
    e_pred = q_pred.mean((1, 2, 3)) + 0.25 * (-pred_out["pair"].sigmoid().squeeze(1))
    e_gt = q_gt.mean((1, 2, 3)) + 0.25 * (-gt_out["pair"].sigmoid().squeeze(1))
    rank = F.softplus(e_pred - e_gt + margin).mean()
    fp = (((1.0 - target) * student_mask * q_pred).sum() /
          ((1.0 - target).sum().clamp_min(1.0)))
    return rank + fp, {"rank": rank.detach(), "fp": fp.detach()}


def _relation_energy(out, pair_weight=0.25):
    """Lower is a more semantically consistent RGB--mask relation."""
    mismatch = out["mismatch"].sigmoid().mean((1, 2, 3))
    invalid_pair = 1.0 - out["pair"].sigmoid().squeeze(1)
    return mismatch + pair_weight * invalid_pair


def oasis_rc_student_loss_v2(pred_out, gt_out, corrupted_out, student_mask,
                             target, margin=0.10, pair_weight=0.25,
                             corrupted_rank_weight=1.0):
    """OASIS-RC-v2 student objective.

    In addition to v1's GT-vs-prediction relation, v2 explicitly requires the
    predicted mask to be more consistent than an online corrupted mask from
    the same image. Critic parameters are frozen by the caller. GT and
    corrupted reference energies are detached; gradients flow only through
    the student's soft prediction and its critic forward pass.
    """
    e_pred = _relation_energy(pred_out, pair_weight)
    e_gt = _relation_energy(gt_out, pair_weight).detach()
    e_corrupted = _relation_energy(corrupted_out, pair_weight).detach()

    rank_gt = F.softplus(e_pred - e_gt + margin).mean()
    rank_corrupted = F.softplus(e_pred - e_corrupted + margin).mean()
    q_pred = pred_out["mismatch"].sigmoid()
    fp = (((1.0 - target) * student_mask * q_pred).sum() /
          ((1.0 - target).sum().clamp_min(1.0)))
    total = rank_gt + corrupted_rank_weight * rank_corrupted + fp
    return total, {
        "rank_gt": rank_gt.detach(),
        "rank_corrupted": rank_corrupted.detach(),
        "fp": fp.detach(),
    }

def oasis_discriminator_loss(logits_real, semantic, logits_fake):
    # Crack pixels are rare; without class balancing this degenerates to background CE.
    weights = logits_real.new_tensor([1.0, 8.0, 1.0])
    real = F.cross_entropy(logits_real, semantic.long(), weight=weights)
    fake_target = torch.full(logits_fake.shape[:1] + logits_fake.shape[2:], 2, device=logits_fake.device, dtype=torch.long)
    fake = F.cross_entropy(logits_fake, fake_target)
    return real + fake

def oasis_generator_loss(logits_fake, desired_semantic):
    return F.cross_entropy(logits_fake, desired_semantic.long())
