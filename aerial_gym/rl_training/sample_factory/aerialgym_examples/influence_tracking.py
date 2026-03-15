from __future__ import annotations

from sample_factory.algo.learning.learner import Learner
from sample_factory.train import run_rl
from sample_factory.utils.typing import Config

from aerial_gym.rl_training.sample_factory.aerialgym_examples.influence_learner_hooks import (
    create_enhanced_learner_init,
    create_enhanced_train,
)
from aerial_gym.rl_training.sample_factory.aerialgym_examples.influence_metric_utils import (
    CURRICULUM_KEYS,
)
from aerial_gym.rl_training.sample_factory.aerialgym_examples.influence_wandb_logging import (
    build_enhanced_wandb_log,
)


def run_with_influence_tracking(cfg: Config) -> None:
    """Enhanced training with complete observation influence tracking."""
    try:
        from aerial_gym.utils.gradient_attribution import (
            GRAD_ATTR_AVAILABLE,
            create_gradient_tracker,
        )
        from aerial_gym.utils.gradient_monitor import (
            INFLUENCE_MONITOR_AVAILABLE,
            create_influence_tracker,
        )
    except ImportError:
        print("Influence/gradient trackers not available")
        INFLUENCE_MONITOR_AVAILABLE = False
        GRAD_ATTR_AVAILABLE = False

    if not INFLUENCE_MONITOR_AVAILABLE:
        print(
            "Complete observation influence tracker not available"
            " - falling back to standard training"
        )
        return run_rl(cfg)

    print("Complete observation influence tracking ENABLED")
    print(f"  Log interval: {cfg.gradient_log_interval} steps")
    print(f"  Print interval: {cfg.gradient_print_interval} steps")

    original_wandb_log = _get_original_wandb_log(cfg)

    tracker_state: dict[str, object] = {"influence": None, "grad": None}
    last_obsgrad_influence: dict[str, float] = {}
    last_obsgrad_grad: dict[str, float] = {}
    last_curriculum: dict[str, float] = {k: 0.0 for k in CURRICULUM_KEYS}

    tracker_config: dict[str, int] = {
        "log_interval": cfg.gradient_log_interval,
        "print_interval": cfg.gradient_print_interval,
    }
    grad_config: dict[str, int] = {
        "log_interval": cfg.gradient_log_interval,
        "print_interval": cfg.gradient_print_interval,
    }

    original_learner_init = Learner.init
    original_learner_train = Learner.train

    if original_wandb_log:
        enhanced_log = build_enhanced_wandb_log(
            cfg=cfg,
            original_wandb_log=original_wandb_log,
            get_influence_tracker=lambda: tracker_state.get("influence"),
            get_grad_tracker=lambda: tracker_state.get("grad"),
            last_obsgrad_influence=last_obsgrad_influence,
            last_obsgrad_grad=last_obsgrad_grad,
        )

    Learner.init = create_enhanced_learner_init(
        original_init=original_learner_init,
        cfg=cfg,
        tracker_config=tracker_config,
        grad_config=grad_config,
        create_influence_tracker=create_influence_tracker,
        create_gradient_tracker=create_gradient_tracker,
        tracker_state=tracker_state,
    )
    Learner.train = create_enhanced_train(
        original_train=original_learner_train,
        cfg=cfg,
        last_curriculum=last_curriculum,
    )

    if original_wandb_log:
        import wandb

        wandb.log = enhanced_log

    try:
        print("Starting enhanced training with observation influence tracking...")
        result = run_rl(cfg)

        _print_final_summary(tracker_state)

        return result
    finally:
        Learner.init = original_learner_init
        Learner.train = original_learner_train
        if original_wandb_log:
            import wandb

            wandb.log = original_wandb_log


def _get_original_wandb_log(cfg: Config) -> object | None:
    """Retrieve the original wandb.log function if wandb is enabled."""
    if not cfg.with_wandb:
        return None
    try:
        import wandb

        return wandb.log
    except ImportError:
        return None


def _print_final_summary(tracker_state: dict[str, object]) -> None:
    """Print final analysis summaries and clean up trackers."""
    influence_tracker = tracker_state.get("influence")
    if influence_tracker:
        print(f"Training completed with {influence_tracker.step_count} analysis steps")
        influence_tracker.print_analysis_summary()
        influence_tracker.cleanup()
    else:
        print("No influence tracker was created - analysis unavailable")

    grad_tracker = tracker_state.get("grad")
    if grad_tracker:
        grad_tracker.print_gradient_summary()
        grad_tracker.cleanup()
    else:
        print("No gradient attribution tracker was created - analysis unavailable")
