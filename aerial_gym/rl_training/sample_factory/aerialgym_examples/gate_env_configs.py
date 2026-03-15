from __future__ import annotations

# custom default configuration parameters for specific envs
# add more envs here analogously (env names should match config file names in IGE)
env_configs = dict(
    position_setpoint_task=dict(
        train_for_env_steps=131000000000,
        encoder_mlp_layers=[256, 128, 64],
        gamma=0.99,
        rollout=16,
        learning_rate=1e-4,
        lr_schedule_kl_threshold=0.016,
        batch_size=16384,
        num_epochs=4,
        max_grad_norm=1.0,
        num_batches_per_epoch=4,
        exploration_loss_coeff=0.0,
        with_wandb=False,
        wandb_project="quad",
        wandb_user="mihirkulkarni",
    ),
    navigation_task=dict(
        train_for_env_steps=131000000000,
        encoder_mlp_layers=[256, 128, 64],
        use_rnn=True,
        encoder_conv_architecture="convnet_simple",  # "resnet_impala_mihirk",
        rnn_num_layers=1,
        rnn_size=64,
        rnn_type="gru",
        gamma=0.98,
        rollout=32,
        learning_rate=1e-4,
        lr_schedule_kl_threshold=0.016,
        batch_size=1024,
        num_epochs=4,
        max_grad_norm=1.0,
        num_batches_per_epoch=4,
        exploration_loss_coeff=0.0,
        with_wandb=False,
        wandb_project="quad",
        wandb_user="mihirkulkarni",
    ),
    quad_with_obstacles=dict(
        # CRITICAL FIX: Configure for 3D action space to match inference expectations
        # The inference script expects 3D actions, so train with 3D to avoid shape mismatch
        adaptive_stddev=True,  # Can use adaptive_stddev with 3D actions
        action_space_dim=3,  # FORCE 3D action space for inference compatibility
        # MODIFIED CONFIGURATION - Single input processing to match old 1333 model
        train_for_env_steps=100000000,  # 100M steps to match original config
        encoder_mlp_layers=[512, 256, 64],  # Match original config
        # REMOVED ConvNet processing - VAE latents handled by DCE task, single input only
        encoder_conv_mlp_layers=[],  # Disable ConvNet encoder for single input processing
        use_rnn=True,
        rnn_num_layers=1,
        rnn_size=64,
        rnn_type="gru",
        recurrence=32,  # Match original config
        gamma=0.98,
        reward_scale=0.1,  # Match original config
        rollout=32,  # REVERTED: Issue was tensor reference bug, not rollout frequency
        learning_rate=0.0003,  # Match original config (3e-4)
        lr_schedule="kl_adaptive_epoch",
        lr_schedule_kl_threshold=0.016,
        # UPDATED CONFIG - 128 environments with optimized batch accumulation
        batch_size=16384,  # 8x batch size for 128 environments (was 2048 for 16 envs)
        num_batches_to_accumulate=1,  # Reduced accumulation for memory optimization
        num_batches_per_epoch=8,  # Keep batches per epoch the same
        num_epochs=4,
        max_grad_norm=1.0,
        exploration_loss_coeff=0.001,  # Match original config
        value_loss_coeff=2.0,  # Match original config
        kl_loss_coeff=0.1,  # Match original config
        normalize_input=True,  # Match original config
        normalize_returns=True,  # Match original config
        async_rl=False,  # Force synchronous training for determinism
        serial_mode=True,  # Match original config
        batched_sampling=True,  # Match original config
        num_workers=1,  # Match original config
        num_envs_per_worker=1,  # Match original config
        # UPDATED CONFIG - 128 environments (3D action space for inference compatibility)
        env_agents=16,  # Default to 16 environments for standard training
        worker_num_splits=1,  # Match original config
        policy_workers_per_policy=1,  # Match original config
        nonlinearity="elu",  # Match original config
        shuffle_minibatches=False,  # Match original config (was True in defaults)
        gae_lambda=0.95,  # Match original config
        ppo_clip_ratio=0.2,  # Match original config
        ppo_clip_value=1.0,  # Match original config
        with_vtrace=False,  # Match original config
        value_bootstrap=True,  # Match original config
        reward_clip=1000.0,  # Match original config
        obs_subtract_mean=0.0,  # Match original config
        obs_scale=1.0,  # Match original config
        decorrelate_experience_max_seconds=0,  # Match original config
        decorrelate_envs_on_one_worker=True,  # Match original config
        max_policy_lag=1000,  # Match original config
        vtrace_rho=1.0,  # Match original config
        vtrace_c=1.0,  # Match original config
        lr_adaptive_min=1e-06,  # Match original config
        lr_adaptive_max=0.01,  # Match original config
        save_every_sec=1800,  # Regular checkpoint every 30 minutes
        keep_checkpoints=5,  # Keep 5 regular checkpoints (increased for safety)
        save_milestones_sec=-1,  # No milestone saving
        save_best_every_sec=300,  # Check for best model every 5 minutes
        save_best_metric="reward",  # Use reward to determine best model
        save_best_after=100000,  # Save best models after 100K steps (much more reasonable)
        policy_initialization="torch_default",  # Match original config
        policy_init_gain=1.0,  # Match original config
        actor_critic_share_weights=True,  # Match original config
        # adaptive_stddev=False set above to prevent 12D action space doubling
        continuous_tanh_scale=0.0,  # Match original config
        initial_stddev=1.0,  # Match original config
        restart_behavior="resume",  # Match original config (not "overwrite")
        optimizer="adam",  # Match original config
        adam_eps=1e-06,  # Match original config
        adam_beta1=0.9,  # Match original config
        adam_beta2=0.999,  # Match original config
        exploration_loss="entropy",  # Match original config
        decoder_mlp_layers=[],  # Match original config
        env_frameskip=1,  # Match original config
        env_framestack=1,  # Match original config
        pixel_format="CHW",  # Match original config
        use_record_episode_statistics=False,  # Match original config
        normalize_input_keys=None,  # Match original config
        set_workers_cpu_affinity=True,  # Match original config
        force_envs_single_thread=False,  # Match original config
        default_niceness=0,  # Match original config
        log_to_file=True,  # Match original config
        experiment_summaries_interval=10,  # Match original config
        flush_summaries_interval=30,  # Match original config
        stats_avg=100,  # Match original config
        summaries_use_frameskip=True,  # Match original config
        heartbeat_interval=20,  # Match original config
        heartbeat_reporting_interval=180,  # Match original config
        train_for_seconds=10000000000,  # Match original config
        load_checkpoint_kind="latest",  # Match original config
        benchmark=False,  # Match original config
        with_wandb=True,  # Enable Weights & Biases logging
        wandb_project="vae_rl_navigation",  # Match original project name
        wandb_user="ziya-ruso-ucl",  # Your team entity name
        wandb_group="dce_navigation_training",
        wandb_tags=["aerial_gym", "dce", "navigation", "sample_factory"],
        wandb_job_type="SF",  # Match original config
        with_pbt=False,  # Match original config
        pbt_mix_policies_in_one_env=True,  # Match original config
        pbt_period_env_steps=5000000,  # Match original config
        pbt_start_mutation=20000000,  # Match original config
        pbt_replace_fraction=0.3,  # Match original config
        pbt_mutation_rate=0.15,  # Match original config
        pbt_replace_reward_gap=0.1,  # Match original config
        pbt_replace_reward_gap_absolute=1e-06,  # Match original config
        pbt_optimize_gamma=False,  # Match original config
        pbt_target_objective="true_objective",  # Match original config
        pbt_perturb_min=1.1,  # Match original config
        pbt_perturb_max=1.5,  # Match original config
        help=False,  # Match original config
        algo="APPO",  # Match original config
        device="gpu",  # Match original config
        seed=None,  # Match original config
        num_policies=1,  # Match original config
        actor_worker_gpus=[0],  # Match original config
        obs_key="obs",  # Match original config
        subtask=None,  # Match original config
        ige_api_version="preview4",  # Match original config
        eval_stats=False,  # Match original config
    ),
    quad_with_obstacles_gate=dict(
        # Gate Navigation Task Configuration
        # PURE VISION NAVIGATION EXPERIMENT: 141D observation space (13D basic + 64D drone VAE + 64D static camera VAE)
        # Removed explicit target guidance to test vision-based navigation through gate
        # X500 robot with D455 camera flying through static gate with static camera
        adaptive_stddev=True,  # Can use adaptive_stddev with 4D actions
        action_space_dim=4,  # 4D action space for VELOCITY CONTROLLER (x_vel, y_vel, z_vel, yaw_rate)
        # Gate Navigation Training Configuration
        train_for_env_steps=200000000,  # 200M steps for comprehensive gate navigation learning
        encoder_mlp_layers=[512, 256, 128],  # Larger network for 145D observation space
        encoder_conv_mlp_layers=[],  # No ConvNet - VAE latents handled by gate task
        use_rnn=True,
        rnn_num_layers=1,
        rnn_size=128,  # Larger RNN for gate navigation complexity
        rnn_type="gru",
        recurrence=32,
        gamma=0.98,
        reward_scale=0.1,
        rollout=32,  # REVERTED: Issue was tensor reference bug, not rollout frequency
        learning_rate=0.0003,
        lr_schedule="kl_adaptive_epoch",
        lr_schedule_kl_threshold=0.016,
        # Optimized for 128 environments with dual camera processing
        batch_size=8192,  # Reduced batch size for 145D observations and dual cameras
        num_batches_to_accumulate=2,  # Accumulate for effective batch size 16384
        num_batches_per_epoch=4,  # Keep same as base DCE
        num_epochs=4,
        max_grad_norm=1.0,
        exploration_loss_coeff=0.001,
        value_loss_coeff=2.0,
        kl_loss_coeff=0.1,
        normalize_input=True,
        normalize_returns=True,
        async_rl=False,
        serial_mode=True,
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        # Gate Navigation Environment Configuration
        env_agents=16,  # Default to 16 environments; can still be overridden via CLI/env
        worker_num_splits=1,
        policy_workers_per_policy=1,
        nonlinearity="elu",
        shuffle_minibatches=False,
        gae_lambda=0.95,
        ppo_clip_ratio=0.2,
        ppo_clip_value=1.0,
        with_vtrace=False,
        value_bootstrap=True,
        reward_clip=1000.0,
        obs_subtract_mean=0.0,
        obs_scale=1.0,
        decorrelate_experience_max_seconds=0,
        decorrelate_envs_on_one_worker=True,
        max_policy_lag=1000,
        vtrace_rho=1.0,
        vtrace_c=1.0,
        lr_adaptive_min=1e-06,
        lr_adaptive_max=0.01,
        # Checkpoint and logging
        save_every_sec=1800,  # Save every 30 minutes
        keep_checkpoints=5,
        save_milestones_sec=-1,
        save_best_every_sec=300,
        save_best_metric="reward",
        save_best_after=100000,  # Save best models after 100K steps
        # Model configuration
        policy_initialization="torch_default",
        policy_init_gain=1.0,
        actor_critic_share_weights=True,
        continuous_tanh_scale=0.0,
        initial_stddev=1.0,
        restart_behavior="resume",
        optimizer="adam",
        adam_eps=1e-06,
        adam_beta1=0.9,
        adam_beta2=0.999,
        exploration_loss="entropy",
        decoder_mlp_layers=[],
        # Environment settings
        env_frameskip=1,
        env_framestack=1,
        pixel_format="CHW",
        use_record_episode_statistics=False,
        normalize_input_keys=None,
        set_workers_cpu_affinity=True,
        force_envs_single_thread=False,
        default_niceness=0,
        # Logging and monitoring
        log_to_file=True,
        experiment_summaries_interval=10,
        flush_summaries_interval=30,
        stats_avg=100,
        summaries_use_frameskip=True,
        heartbeat_interval=20,
        heartbeat_reporting_interval=180,
        train_for_seconds=10000000000,
        load_checkpoint_kind="latest",
        benchmark=False,
        # Weights & Biases logging
        with_wandb=True,
        wandb_project="gate_navigation_dual_camera",  # New project for gate navigation
        wandb_user="ziya-ruso-ucl",
        wandb_group="gate_navigation_training",
        wandb_tags=["aerial_gym", "gate_navigation", "dual_camera", "x500", "sample_factory"],
        wandb_job_type="SF",
        # Population Based Training (disabled for now)
        with_pbt=False,
        pbt_mix_policies_in_one_env=True,
        pbt_period_env_steps=5000000,
        pbt_start_mutation=20000000,
        pbt_replace_fraction=0.3,
        pbt_mutation_rate=0.15,
        pbt_replace_reward_gap=0.1,
        pbt_replace_reward_gap_absolute=1e-06,
        pbt_optimize_gamma=False,
        pbt_target_objective="true_objective",
        pbt_perturb_min=1.1,
        pbt_perturb_max=1.5,
        # Sample Factory specific
        help=False,
        algo="APPO",
        device="gpu",
        seed=None,
        num_policies=1,
        actor_worker_gpus=[0],
        obs_key="obs",
        subtask=None,
        ige_api_version="preview4",
        eval_stats=False,
    ),
)
