"""Source-anchored canonical OASIS-RC v2 entrypoint for implementation 2.0.3."""

from . import train_oasis_rc_v2_legacy as _legacy
from .critic_contract_v202 import validate_loaded_critic as _validate_strict
from .train_critic_v202 import train_critic

AOSK_VARIANT = "oriented-consistency-v1"

augment = _legacy.augment
build_targets = _legacy.build_targets
configure_determinism = _legacy.configure_determinism
critic_metrics = _legacy.critic_metrics
load_student_init = _legacy.load_student_init
make_corrupted_mask = _legacy.make_corrupted_mask
make_generator = _legacy.make_generator
make_loader = _legacy.make_loader
make_student = _legacy.make_student
make_train_loader = _legacy.make_train_loader
manifest_has_split = _legacy.manifest_has_split
manifest_splits = _legacy.manifest_splits
runtime_metadata = _legacy.runtime_metadata
seed_all = _legacy.seed_all
segmentation_metrics = _legacy.segmentation_metrics
select_threshold = _legacy.select_threshold
sha256_file = _legacy.sha256_file
threshold_sweep_metrics = _legacy.threshold_sweep_metrics
type_name_for_student = _legacy.type_name_for_student

validate_loaded_critic = _validate_strict


def main():
    _legacy.train_critic = train_critic
    _legacy.validate_loaded_critic = _validate_strict
    _legacy.AOSK_VARIANT = AOSK_VARIANT
    return _legacy.main()


if __name__ == "__main__":
    main()
