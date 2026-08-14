from oasis_rc_v2.checkpoint import validate_critic_checkpoint


def training_hparams(args, cfg, determinism_mode):
    return {
        "lr": float(args.lr),
        "critic_epochs": int(args.critic_epochs),
        "critic_width": int(args.critic_width),
        "batch_size": int(cfg["batch_size"]),
        "crack_dice_weight": float(args.crack_dice_weight),
        "mismatch_weight": float(args.mismatch_weight),
        "pair_weight": float(args.pair_weight),
        "rgb_mask_weight": float(args.rgb_mask_weight),
        "normal_critic_weight": float(args.normal_critic_weight),
        "normal_fraction": float(args.normal_fraction),
        "determinism_mode": determinism_mode,
        "rgb_shuffle_pair_only": True,
        "mask_flip_training": False,
        "mask_variant_contract": "operator-preserved-v1",
    }


def validate_loaded_critic(saved, args, cfg):
    expected = {
        "crack_dice_weight": float(args.crack_dice_weight),
        "mismatch_weight": float(args.mismatch_weight),
        "pair_weight": float(args.pair_weight),
        "rgb_mask_weight": float(args.rgb_mask_weight),
        "normal_critic_weight": float(args.normal_critic_weight),
        "normal_fraction": float(args.normal_fraction),
        "rgb_shuffle_pair_only": True,
        "mask_flip_training": False,
        "mask_variant_contract": "operator-preserved-v1",
    }
    return validate_critic_checkpoint(
        saved,
        args.manifest,
        cfg,
        args.normal_fraction,
        args.normal_critic_weight,
        dataset_content_sha256_value=args._dataset_content_sha256,
        expected_hparams=expected,
    )
