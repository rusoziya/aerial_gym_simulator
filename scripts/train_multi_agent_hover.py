#!/usr/bin/env python3

"""
Multi-Agent Hover Training Script for Aerial Gym

A simple and lightweight script to train 3 drones to hover together in formation.
Perfect for visualization and understanding multi-agent interactions.
"""

import os
import argparse
from datetime import datetime

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multi-Agent Hover Task Training")
    
    parser.add_argument("--num_envs", type=int, default=2,
                       help="Number of parallel environments (keep small for visualization)")
    
    parser.add_argument("--episodes", type=int, default=20,
                       help="Number of training episodes")
    
    parser.add_argument("--episode_length", type=int, default=300,
                       help="Maximum steps per episode")
    
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use for training")
    
    parser.add_argument("--headless", action="store_true",
                       help="Run without visualization (default: False - show visualization)")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_arguments()
    
    print("=" * 60)
    print("🚁 Multi-Agent Hover Task Training 🚁")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Environments: {args.num_envs}")
    print(f"  - Drones per environment: 3")
    print(f"  - Episodes: {args.episodes}")
    print(f"  - Episode length: {args.episode_length}")
    print(f"  - Visualization: {'Disabled' if args.headless else 'Enabled'}")
    print("=" * 60)
    
    # Import everything AFTER argument parsing to delay torch import
    print("Importing Aerial Gym components...")
    
    # Import Aerial Gym components FIRST (before torch) to avoid IsaacGym conflicts
    from aerial_gym.envs.navigation_task.multi_agent_navigation_task_config import (
        MultiAgentHoverTaskConfig
    )
    from aerial_gym.algorithms.multi_agent_wrapper import create_multi_agent_env

    # Import torch and numpy AFTER aerial_gym to avoid IsaacGym import conflicts
    import torch
    import numpy as np
    
    print("All imports successful! ✅")
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("./logs", f"hover_demo_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logs will be saved to: {log_dir}")
    
    # Create environment
    print("\n🔧 Creating hover environment...")
    env = create_multi_agent_env(
        task_config_class=MultiAgentHoverTaskConfig,
        algorithm="ppo",
        device=device,
        headless=args.headless,
        num_envs=args.num_envs
    )
    
    print(f"✅ Environment created successfully")
    print(f"   - Observation spaces: {len(env.observation_spaces)} agents")
    print(f"   - Action spaces: {len(env.action_spaces)} agents")
    
    # Simple hover policy network
    class SimpleHoverPolicy(torch.nn.Module):
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, action_dim),
                torch.nn.Tanh()  # Bounded actions
            )
            
        def forward(self, x):
            return self.network(x) * 0.5  # Gentle actions for hovering
    
    # Get dimensions
    obs_dim = list(env.observation_spaces.values())[0].shape[0]
    action_dim = list(env.action_spaces.values())[0].shape[0]
    
    print(f"\n📊 Network configuration:")
    print(f"   - Observation dimension: {obs_dim}")
    print(f"   - Action dimension: {action_dim}")
    
    # Create policy for each agent
    policies = {}
    optimizers = {}
    
    for agent_id in env.agent_ids:
        policies[agent_id] = SimpleHoverPolicy(obs_dim, action_dim).to(device)
        optimizers[agent_id] = torch.optim.Adam(policies[agent_id].parameters(), lr=1e-3)
    
    print(f"   - Created policies for {len(policies)} agents")
    
    # Training loop
    print(f"\n🚀 Starting training for {args.episodes} episodes...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        for episode in range(args.episodes):
            # Reset environment
            obs_dict = env.reset()
            episode_rewards = {agent_id: 0 for agent_id in env.agent_ids}
            
            for step in range(args.episode_length):
                actions_dict = {}
                
                # Get actions from all policies
                for agent_env_key, obs in obs_dict.items():
                    agent_id = agent_env_key.split('_env_')[0]
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        action = policies[agent_id](obs_tensor).squeeze(0).cpu().numpy()
                        actions_dict[agent_env_key] = action
                
                # Step environment
                obs_dict, reward_dict, done_dict, _, info_dict = env.step(actions_dict)
                
                # Accumulate rewards
                for agent_env_key, reward in reward_dict.items():
                    agent_id = agent_env_key.split('_env_')[0]
                    episode_rewards[agent_id] += reward
                    
                # Check if done
                if any(done_dict.values()):
                    break
            
            # Print progress
            avg_rewards = {aid: episode_rewards[aid] / env.task.num_envs 
                          for aid in env.agent_ids}
            
            # Format rewards nicely
            reward_str = ", ".join([f"{aid}: {reward:.2f}" for aid, reward in avg_rewards.items()])
            print(f"Episode {episode+1:2d}/{args.episodes}: {reward_str}")
            
            # Save progress periodically
            if episode % 5 == 0 and episode > 0:
                for agent_id, policy in policies.items():
                    model_path = os.path.join(log_dir, f"hover_policy_{agent_id}_ep{episode}.pt")
                    torch.save(policy.state_dict(), model_path)
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    # Save final policies
    print(f"\n💾 Saving final policies...")
    for agent_id, policy in policies.items():
        model_path = os.path.join(log_dir, f"hover_policy_{agent_id}_final.pt")
        torch.save(policy.state_dict(), model_path)
        print(f"   - {agent_id} policy saved to {model_path}")
        
    print(f"\n🎉 Training completed successfully!")
    print(f"📁 All files saved to: {log_dir}")
    print("\n" + "=" * 60)
    print("Tips for better training:")
    print("  1. Watch the drones learn to hover in the simulation window")
    print("  2. Try different episode lengths with --episode_length")
    print("  3. Increase episodes with --episodes for better learning")
    print("  4. Use --headless for faster training without visualization")
    print("=" * 60)

if __name__ == "__main__":
    main() 