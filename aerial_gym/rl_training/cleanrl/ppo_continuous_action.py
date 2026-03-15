from __future__ import annotations

# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy
import random
import time

import isaacgym  # noqa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from aerial_gym.registry.task_registry import task_registry
from aerial_gym.rl_training.cleanrl.ppo_utils import (
    RecordEpisodeStatisticsTorch,
    get_args,
    layer_init,
)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.task_config.observation_space_dim).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.task_config.observation_space_dim).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.task_config.action_space_dim)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.task_config.action_space_dim)))

    def get_value(self, x) -> None:
        return self.critic(x)

    def get_action_and_value(self, x, action=None) -> None:
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


if __name__ == "__main__":
    args = get_args()

    run_name = f"{args.task}__{args.experiment_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = args.sim_device
    print("using device:", device)

    # env setup
    envs = task_registry.make_task(task_name=args.task)

    envs = RecordEpisodeStatisticsTorch(envs, device)

    print("num actions: ", envs.task_config.action_space_dim)
    print("num obs: ", envs.task_config.observation_space_dim)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.play and args.checkpoint is None:
        raise ValueError("No checkpoint provided for testing.")

    # load checkpoint if needed
    if args.checkpoint is not None:
        print("Loading checkpoint...")
        checkpoint = torch.load(args.checkpoint)
        agent.load_state_dict(checkpoint)
        print("Loaded checkpoint")

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs, envs.task_config.observation_space_dim),
        dtype=torch.float,
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs, envs.task_config.action_space_dim),
        dtype=torch.float,
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _, _, _, _info = envs.reset()
    next_obs = next_obs["observations"]
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)
    num_updates = args.total_timesteps // args.batch_size

    if not args.play:
        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards[step], next_done, info = envs.step(action)
                next_obs = next_obs["observations"]
                if 0 <= step <= 2:
                    for idx, d in enumerate(next_done):
                        if d:
                            episodic_return = info["r"][idx].item()
                            print(f"global_step={global_step}, episodic_return={episodic_return}")
                            writer.add_scalar(
                                "charts/episodic_return", episodic_return, global_step
                            )
                            writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
                            if "consecutive_successes" in info:  # ShadowHand and AllegroHand metric
                                writer.add_scalar(
                                    "charts/consecutive_successes",
                                    info["consecutive_successes"].item(),
                                    global_step,
                                )
                            break

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1, envs.task_config.observation_space_dim))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, envs.task_config.action_space_dim))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            clipfracs = []
            for epoch in range(args.update_epochs):
                b_inds = torch.randperm(args.batch_size, device=device)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )

            # save the model levery 50 updates
            if update % 50 == 0:
                print("Saving model.")
                torch.save(agent.state_dict(), f"runs/{run_name}/latest_model.pth")

    else:
        for step in range(0, 5000000):
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            next_obs, rewards, next_done, info = envs.step(action)

    writer.close()
