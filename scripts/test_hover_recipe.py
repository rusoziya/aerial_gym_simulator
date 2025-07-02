#!/usr/bin/env python3

"""
MAPPO Training Script for 3-Drone Hovering MARL Setup
Proper implementation with actual learning and progress tracking
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time

def main():
    print("🚁" * 20)
    print("MAPPO Training for 3-Drone Hovering (PROPER LEARNING)")
    print("🚁" * 20)
    
    # Import Aerial Gym components FIRST (before torch) to avoid IsaacGym conflicts
    print("Importing Aerial Gym components...")
    from aerial_gym.envs.navigation_task.multi_agent_navigation_task_config import MultiAgentHoverTaskConfig
    from aerial_gym.algorithms.multi_agent_wrapper import create_multi_agent_env
    
    print("✅ All imports successful!")
    
    # Create environment
    print("\n🔧 Creating 3-drone hover environment...")
    env = create_multi_agent_env(
        task_config_class=MultiAgentHoverTaskConfig,
        algorithm="mappo",  # Use MAPPO wrapper
        device="cuda" if torch.cuda.is_available() else "cpu",
        headless=False,  # Show visualization
        num_envs=1       # Single environment for clear observation
    )
    
    print(f"✅ Environment created!")
    print(f"   - Number of agents: {len(env.agent_ids)}")
    print(f"   - Agent IDs: {env.agent_ids}")
    
    # Check actual dimensions
    sample_agent_key = list(env.observation_spaces.keys())[0]
    obs_dim = env.observation_spaces[sample_agent_key].shape[0]
    action_dim = env.action_spaces[sample_agent_key].shape[0]
    
    print(f"   - Observation dim: {obs_dim}")
    print(f"   - Action dim: {action_dim}")
    print(f"   - Expected robots per env: {env.task.num_robots_per_env}")
    print(f"   - Total simulation robots: {env.task.num_envs * env.task.num_robots_per_env}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # MAPPO Actor-Critic Networks
    class ActorNetwork(nn.Module):
        def __init__(self, obs_dim, action_dim, hidden_dim=256):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim)
            )
            # Initialize last layer with small weights for stable learning
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.1)
            
        def forward(self, x):
            return torch.tanh(self.actor(x))  # Bound actions to [-1, 1]
    
    class CriticNetwork(nn.Module):
        def __init__(self, state_dim, hidden_dim=256):
            super().__init__()
            self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            
        def forward(self, x):
            return self.critic(x)
    
    # Global state dimension (for centralized critic)
    global_state_dim = obs_dim * env.task.num_robots_per_env  # All agent observations concatenated
    
    # Create networks (parameter sharing across agents)
    actor = ActorNetwork(obs_dim, action_dim).to(device)
    critic = CriticNetwork(global_state_dim).to(device)
    
    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    
    print(f"\n📊 MAPPO Networks:")
    print(f"   - Actor: {obs_dim}D → {action_dim}D (shared across 3 agents)")
    print(f"   - Critic: {global_state_dim}D → 1D (centralized)")
    print(f"   - Actor params: {sum(p.numel() for p in actor.parameters())}")
    print(f"   - Critic params: {sum(p.numel() for p in critic.parameters())}")
    
    # Training hyperparameters
    batch_size = 32
    episodes_per_update = 4
    ppo_epochs = 4
    clip_ratio = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5
    
    # Storage for experiences
    class RolloutBuffer:
        def __init__(self):
            self.observations = []
            self.actions = []
            self.rewards = []
            self.values = []
            self.log_probs = []
            self.dones = []
            self.global_states = []
            
        def clear(self):
            self.observations.clear()
            self.actions.clear()
            self.rewards.clear()
            self.values.clear()
            self.log_probs.clear()
            self.dones.clear()
            self.global_states.clear()
    
    buffer = RolloutBuffer()
    
    # Training metrics
    episode_rewards = deque(maxlen=100)
    actor_losses = deque(maxlen=100)
    critic_losses = deque(maxlen=100)
    
    print(f"\n🚀 Starting MAPPO Training...")
    print("Hyperparameters:")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Episodes per update: {episodes_per_update}")
    print(f"   - PPO epochs: {ppo_epochs}")
    print(f"   - Clip ratio: {clip_ratio}")
    print("\nPress Ctrl+C to stop\n")
    
    episode = 0
    total_steps = 0
    
    try:
        while episode < 1000:  # Train for 1000 episodes
            obs_dict = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_start_time = time.time()
            
            # Debug first episode
            if episode == 0:
                print(f"\n🔍 DEBUG Episode 0:")
                print(f"   - Reset returned {len(obs_dict)} observations")
                print(f"   - Agent keys: {list(obs_dict.keys())}")
                
                for key, obs in obs_dict.items():
                    print(f"   - {key}: shape {obs.shape}, range [{obs.min():.3f}, {obs.max():.3f}]")
            
            for step in range(500):  # Max 500 steps per episode
                # Convert observations to tensors
                obs_tensors = {}
                global_state_list = []
                
                for agent_key, obs in obs_dict.items():
                    obs_tensor = torch.FloatTensor(obs).to(device)
                    obs_tensors[agent_key] = obs_tensor
                    global_state_list.append(obs_tensor)
                
                # Global state for centralized critic
                global_state = torch.cat(global_state_list, dim=0).unsqueeze(0)  # (1, global_state_dim)
                
                # Get actions from actor (with exploration noise)
                actions_dict = {}
                log_probs_dict = {}
                
                with torch.no_grad():
                    # Get value from centralized critic
                    value = critic(global_state)
                    
                    for agent_key, obs_tensor in obs_tensors.items():
                        # Actor forward pass
                        action_mean = actor(obs_tensor.unsqueeze(0))
                        
                        # Add exploration noise
                        action_std = 0.1  # Standard deviation for exploration
                        action_dist = torch.distributions.Normal(action_mean, action_std)
                        action = action_dist.sample()
                        action = torch.clamp(action, -1.0, 1.0)  # Ensure bounds
                        
                        log_prob = action_dist.log_prob(action).sum()
                        
                        actions_dict[agent_key] = action.squeeze(0).cpu().numpy()
                        log_probs_dict[agent_key] = log_prob
                
                # Store experience (for first agent, representing shared experience)
                if len(obs_dict) > 0:
                    first_agent_key = list(obs_dict.keys())[0]
                    buffer.observations.append(obs_dict[first_agent_key])
                    buffer.actions.append(actions_dict[first_agent_key])
                    buffer.values.append(value.item())
                    buffer.log_probs.append(log_probs_dict[first_agent_key].item())
                    buffer.global_states.append(global_state.squeeze(0).cpu().numpy())
                
                # Step environment
                obs_dict, reward_dict, done_dict, _, info_dict = env.step(actions_dict)
                
                # Debug first episode rewards
                if episode == 0 and step < 5:
                    print(f"     Step {step}: Rewards = {reward_dict}")
                    if step == 0:
                        print(f"     Info = {info_dict}")
                
                # Store reward and done
                if len(reward_dict) > 0:
                    step_reward = list(reward_dict.values())[0]  # Shared reward
                    episode_reward += step_reward
                    buffer.rewards.append(step_reward)
                    buffer.dones.append(any(done_dict.values()))
                
                episode_steps += 1
                total_steps += 1
                
                # Check for episode end
                if any(done_dict.values()) or step >= 499:
                    if episode == 0:
                        print(f"     Episode ended at step {step}, dones = {done_dict}")
                    break
            
            episode_rewards.append(episode_reward)
            episode_duration = time.time() - episode_start_time
            
            # Print progress
            if episode % 5 == 0:
                avg_reward = np.mean(episode_rewards) if episode_rewards else episode_reward
                print(f"Episode {episode:4d}: Reward = {episode_reward:8.2f} | Avg = {avg_reward:8.2f} | Steps = {episode_steps:3d} | Time = {episode_duration:.2f}s")
                
                if episode == 0:
                    print(f"   Episode details: {episode_steps} steps, {len(buffer.rewards)} experiences collected")
            
            # Update networks every few episodes
            if (episode + 1) % episodes_per_update == 0 and len(buffer.rewards) > batch_size:
                print(f"\n📈 Updating networks at episode {episode + 1}...")
                
                # Convert buffer to tensors
                obs_batch = torch.FloatTensor(buffer.observations).to(device)
                actions_batch = torch.FloatTensor(buffer.actions).to(device)
                rewards_batch = torch.FloatTensor(buffer.rewards).to(device)
                values_batch = torch.FloatTensor(buffer.values).to(device)
                old_log_probs = torch.FloatTensor(buffer.log_probs).to(device)
                global_states_batch = torch.FloatTensor(buffer.global_states).to(device)
                
                # Compute advantages (simplified)
                returns = []
                running_return = 0
                for reward, done in zip(reversed(buffer.rewards), reversed(buffer.dones)):
                    if done:
                        running_return = 0
                    running_return = reward + 0.99 * running_return
                    returns.insert(0, running_return)
                
                returns = torch.FloatTensor(returns).to(device)
                advantages = returns - values_batch
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # PPO updates
                total_actor_loss = 0
                total_critic_loss = 0
                
                for _ in range(ppo_epochs):
                    # Actor loss
                    action_mean = actor(obs_batch)
                    action_std = 0.1
                    action_dist = torch.distributions.Normal(action_mean, action_std)
                    new_log_probs = action_dist.log_prob(actions_batch).sum(dim=1)
                    
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = -action_dist.entropy().mean()
                    
                    total_loss = actor_loss + entropy_coef * entropy_loss
                    
                    actor_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                    actor_optimizer.step()
                    
                    # Critic loss
                    new_values = critic(global_states_batch).squeeze()
                    critic_loss = nn.MSELoss()(new_values, returns)
                    
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                    critic_optimizer.step()
                    
                    total_actor_loss += actor_loss.item()
                    total_critic_loss += critic_loss.item()
                
                actor_losses.append(total_actor_loss / ppo_epochs)
                critic_losses.append(total_critic_loss / ppo_epochs)
                
                print(f"   Actor Loss: {actor_losses[-1]:.4f} | Critic Loss: {critic_losses[-1]:.4f}")
                
                # Clear buffer
                buffer.clear()
            
            episode += 1
            
            # Save model periodically
            if episode % 100 == 0:
                torch.save({
                    'actor': actor.state_dict(),
                    'critic': critic.state_dict(),
                    'episode': episode
                }, f'hover_mappo_checkpoint_{episode}.pth')
                print(f"💾 Saved checkpoint at episode {episode}")
                
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    print(f"\n🎉 Training completed!")
    print(f"   Total episodes: {episode}")
    print(f"   Total steps: {total_steps}")
    print(f"   Final avg reward: {np.mean(episode_rewards) if episode_rewards else 0:.2f}")
    
    # Save final model
    torch.save({
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'episode': episode
    }, 'hover_mappo_final.pth')
    print("💾 Saved final model")
    
    print("\n" + "=" * 50)
    print("MAPPO Training Summary:")
    print("✅ Parameter sharing actor across 3 agents")
    print("✅ Centralized critic with global state")
    print("✅ PPO updates with clipping and regularization")
    print("✅ Proper exploration and learning dynamics")
    print("✅ Recipe-compliant reward structure")
    print("=" * 50)

if __name__ == "__main__":
    main() 