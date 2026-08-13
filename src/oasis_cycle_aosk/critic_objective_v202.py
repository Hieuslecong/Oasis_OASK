import torch

from oasis_rc_v2 import build_targets, make_corrupted_mask
from oasis_rc_v2.losses import oasis_rc_critic_loss


def critic_term(critic, image, mask, invalid, args):
    semantic, mismatch, pair_valid = build_targets(mask, invalid)
    return oasis_rc_critic_loss(
        critic(image, mask),
        semantic,
        mismatch,
        pair_valid,
        crack_dice_weight=args.crack_dice_weight,
        mismatch_weight=args.mismatch_weight,
        pair_weight=args.pair_weight,
    )[0]


def batch_objective(critic, image, mask, is_normal, args, generator):
    variant, invalid, meta = make_corrupted_mask(
        mask,
        true_normal=is_normal,
        generator=generator,
        return_meta=True,
        image=image,
    )
    loss = 0.5 * (
        critic_term(critic, image, mask, torch.zeros_like(mask), args)
        + critic_term(critic, image, variant, invalid, args)
    )

    crack = (~is_normal) & (mask.flatten(1).sum(1) > 0)
    rgb_samples = 0
    if crack.any():
        xc, yc = image[crack], mask[crack]
        semantic, mismatch, pair_valid = build_targets(yc, torch.zeros_like(yc))
        rgb_pair_only = oasis_rc_critic_loss(
            critic(xc.flip(-1), yc),
            semantic,
            mismatch,
            torch.zeros_like(pair_valid),
            crack_dice_weight=args.crack_dice_weight,
            mismatch_weight=args.mismatch_weight,
            pair_weight=args.pair_weight,
        )[0]
        loss = loss + args.rgb_mask_weight * rgb_pair_only
        rgb_samples = int(yc.shape[0])

    if is_normal.any() and crack.any():
        normal_rgb = image[is_normal]
        crack_masks = mask[crack]
        ids = torch.randint(
            0,
            crack_masks.shape[0],
            (normal_rgb.shape[0],),
            device=mask.device,
            generator=generator,
        )
        donor = crack_masks[ids]
        loss = loss + args.normal_critic_weight * critic_term(
            critic, normal_rgb, donor, donor.clone(), args
        )

    return loss, meta, rgb_samples
