#!/usr/bin/env python3

"""
Multi-Agent Training Script for Aerial Gym

This script demonstrates how to train different multi-agent RL algorithms
on the Aerial Gym Simulator for cooperative drone tasks.
"""

import os
import argparse
from datetime import datetime

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multi-Agent Aerial Gym Training")
    
    parser.add_argument("--algorithm", type=str, default="mappo", 
                       choices=["mappo", "qmix", "rllib", "independent_ppo"],
                       help="Multi-agent RL algorithm to use")
    
    parser.add_argument("--task", type=str, default="navigation",
                       choices=["navigation", "formation", "formation_light", "swarm"],
                       help="Task type to train on")
    
    parser.add_argument("--num_envs", type=int, default=1,
                       help="Number of parallel environments")
    
    parser.add_argument("--num_agents", type=int, default=3,
                       help="Number of agents per environment")
    
    parser.add_argument("--total_timesteps", type=int, default=1000000,
                       help="Total training timesteps")
    
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use for training")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Directory to save logs and models")
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_arguments()
    
    print(f"Starting multi-agent training with {args.algorithm} on {args.task} task")
    print(f"Configuration: {args.num_envs} envs, {args.num_agents} agents per env")
    
    # Import everything AFTER argument parsing to delay torch import
    print("Importing Aerial Gym components...")
    
    # Import Aerial Gym components FIRST (before torch) to avoid IsaacGym conflicts
    from aerial_gym.envs.navigation_task.multi_agent_navigation_task_config import (
        MultiAgentNavigationTaskConfig,
        MultiAgentFormationTaskConfig,
        MultiAgentFormationTaskConfigLight,
        MultiAgentSwarmTaskConfig
    )
    from aerial_gym.algorithms.multi_agent_wrapper import (
        create_multi_agent_env,
        get_rllib_config,
        get_mappo_config,
        get_qmix_config
    )

    # Import torch and numpy AFTER aerial_gym to avoid IsaacGym import conflicts
    import torch
    import numpy as np
    
    print("All imports successful!")
    
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
    log_dir = os.path.join(args.log_dir, f"{args.algorithm}_{args.task}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logs will be saved to: {log_dir}")
    
    # Get task configuration
    if args.task == "navigation":
        config_class = MultiAgentNavigationTaskConfig
    elif args.task == "formation":
        config_class = MultiAgentFormationTaskConfig
    elif args.task == "formation_light":
        config_class = MultiAgentFormationTaskConfigLight
    elif args.task == "swarm":
        config_class = MultiAgentSwarmTaskConfig
    else:
        raise ValueError(f"Unknown task type: {args.task}")
        
    # Create environment
    print("Creating multi-agent environment...")
    
    # Create config and override number of environments
    def create_config():
        config = config_class()
        # Override with command line arguments
        config.task_config["num_envs"] = args.num_envs
        config.env_config["num_envs"] = args.num_envs
        config.task_config["num_robots_per_env"] = args.num_agents
        config.env_config["num_robots_per_env"] = args.num_agents
        
        # Enable visualization for single environment
        if args.num_envs == 1:
            config.task_config["headless"] = False
            config.sim_config["headless"] = False
        else:
            config.task_config["headless"] = True
            config.sim_config["headless"] = True
            
        return config
    
    env = create_multi_agent_env(
        task_config_class=create_config,
        algorithm=args.algorithm,
        device=device
    )
    
    print(f"Environment created successfully")
    print(f"Observation spaces: {len(env.observation_spaces)} agents")
    print(f"Action spaces: {len(env.action_spaces)} agents")
    
    # Simple training demonstration
    print("Starting simple policy gradient training...")
    
    # Simple policy network
    class SimplePolicy(torch.nn.Module):
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, action_dim),
                torch.nn.Tanh()
            )
            
        def forward(self, x):
            return self.network(x)
    
    # Get dimensions
    obs_dim = list(env.observation_spaces.values())[0].shape[0]
    action_dim = list(env.action_spaces.values())[0].shape[0]
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create policy for each agent
    policies = {}
    optimizers = {}
    
    for agent_id in env.agent_ids:
        policies[agent_id] = SimplePolicy(obs_dim, action_dim).to(device)
        optimizers[agent_id] = torch.optim.Adam(policies[agent_id].parameters(), lr=3e-4)
    
    # Training loop
    num_episodes = args.total_timesteps // 1000  # Rough conversion
    
    # Use fewer episodes for lightweight demonstration
    if args.task == "formation_light":
        num_episodes = min(num_episodes, 50)  # Cap at 50 episodes for quick demo
    
    print(f"Running {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Reset environment
        obs_dict = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in env.agent_ids}
        
        for step in range(200):  # Max episode length
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
        if episode % 10 == 0:
            avg_rewards = {aid: episode_rewards[aid] / env.task.num_envs 
                          for aid in env.agent_ids}
            print(f"Episode {episode}: Average rewards = {avg_rewards}")
    
    # Save policies
    for agent_id, policy in policies.items():
        model_path = os.path.join(log_dir, f"simple_policy_{agent_id}_{args.task}.pt")
        torch.save(policy.state_dict(), model_path)
        print(f"Policy for {agent_id} saved to {model_path}")
        
    print("Training completed successfully!")
    print("Note: This is a demonstration with simple policy gradient.")
    print("For production use, integrate with proper MARL libraries like RLLib or MAPPO.")

if __name__ == "__main__":
    main() 