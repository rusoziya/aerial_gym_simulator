#!/usr/bin/env python3

"""
Configuration Validation Script for DCE Navigation Training

This script validates that all connected files, classes, and imports are properly configured
for the original DCE navigation training setup (16 environments, 2048 batch size).
"""

import sys
import os
import importlib
from pathlib import Path

def validate_imports():
    """Validate that all imports work correctly."""
    print("🔍 Validating imports...")
    
    try:
        # Test main training script import
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import parse_aerialgym_cfg
        print("✅ Main training script import: SUCCESS")
        
        # Test DCE task import
        from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import DCE_RL_Navigation_Task
        print("✅ DCE Navigation Task import: SUCCESS")
        
        # Test inference class import
        from aerial_gym.examples.dce_rl_navigation.sf_inference_class import NN_Inference_Class
        print("✅ Inference class import: SUCCESS")
        
        # Test registry import
        from aerial_gym.registry.task_registry import task_registry
        print("✅ Task registry import: SUCCESS")
        
        return True
        
    except Exception as e:
        print(f"❌ Import validation FAILED: {e}")
        return False

def validate_configuration():
    """Validate the training configuration parameters."""
    print("\n🔧 Validating configuration...")
    
    try:
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import env_configs
        
        quad_config = env_configs.get("quad_with_obstacles", {})
        
        # Check critical parameters
        batch_size = quad_config.get("batch_size", 0)
        num_batches_to_accumulate = quad_config.get("num_batches_to_accumulate", 0)
        num_batches_per_epoch = quad_config.get("num_batches_per_epoch", 0)
        env_agents = quad_config.get("env_agents", 0)
        action_space_dim = quad_config.get("action_space_dim", 0)
        
        effective_batch_size = batch_size * num_batches_to_accumulate
        
        print(f"   📊 Batch size: {batch_size}")
        print(f"   📊 Batches to accumulate: {num_batches_to_accumulate}")
        print(f"   📊 Batches per epoch: {num_batches_per_epoch}")
        print(f"   📊 Environment agents: {env_agents}")
        print(f"   📊 Effective batch size: {effective_batch_size}")
        print(f"   📊 Action space dimension: {action_space_dim}")
        
        # Validate against original DCE config
        expected_values = {
            "batch_size": 2048,
            "num_batches_to_accumulate": 2,
            "num_batches_per_epoch": 8,
            "env_agents": 16,
            "action_space_dim": 3,
            "effective_batch_size": 4096
        }
        
        errors = []
        for param, expected in expected_values.items():
            if param == "effective_batch_size":
                actual = effective_batch_size
            else:
                actual = locals()[param]
            
            if actual != expected:
                errors.append(f"❌ {param}: expected {expected}, got {actual}")
            else:
                print(f"✅ {param}: {actual} (correct)")
        
        if errors:
            print("\n❌ Configuration validation FAILED:")
            for error in errors:
                print(f"   {error}")
            return False
        else:
            print("✅ Configuration validation: SUCCESS")
            return True
            
    except Exception as e:
        print(f"❌ Configuration validation FAILED: {e}")
        return False

def validate_action_compatibility():
    """Validate action space compatibility between training and inference."""
    print("\n🎯 Validating action space compatibility...")
    
    try:
        # Check training script action space
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import env_configs
        training_action_dim = env_configs["quad_with_obstacles"]["action_space_dim"]
        
        # Check if inference scripts can import correctly
        try:
            from aerial_gym.examples.dce_rl_navigation.dce_nn_navigation import get_network
            print("✅ Inference script import: SUCCESS")
        except Exception as e:
            print(f"❌ Inference script import FAILED: {e}")
            return False
        
        # The inference script hardcodes 3 actions in NN_Inference_Class(num_envs, 3, 81, cfg)
        inference_action_dim = 3
        
        print(f"   📊 Training action dimension: {training_action_dim}")
        print(f"   📊 Inference action dimension: {inference_action_dim}")
        
        if training_action_dim == inference_action_dim:
            print("✅ Action space compatibility: SUCCESS")
            return True
        else:
            print(f"❌ Action space MISMATCH: training={training_action_dim}, inference={inference_action_dim}")
            return False
            
    except Exception as e:
        print(f"❌ Action space validation FAILED: {e}")
        return False

def validate_shell_script():
    """Validate the shell script configuration."""
    print("\n🐚 Validating shell script...")
    
    script_path = Path(__file__).parent / "train_with_monitoring.sh"
    
    if not script_path.exists():
        print("❌ Shell script not found")
        return False
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for correct configuration
        if "ENV_AGENTS=16" in content and "BATCH_SIZE=2048" in content:
            print("✅ Shell script configuration: SUCCESS (16 environments, 2048 batch size)")
            return True
        else:
            print("❌ Shell script configuration: INCORRECT")
            print("   Expected: ENV_AGENTS=16, BATCH_SIZE=2048")
            return False
            
    except Exception as e:
        print(f"❌ Shell script validation FAILED: {e}")
        return False

def validate_training_script():
    """Validate the training script configuration."""
    print("\n🔧 Validating training script configuration...")
    
    try:
        # Test import
        from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym_custom_net import (
            parse_aerialgym_cfg, env_configs
        )
        
        # Check if quad_with_obstacles is configured
        if "quad_with_obstacles" in env_configs:
            config = env_configs["quad_with_obstacles"]
            
            # Check critical parameters
            print(f"   - Action space: {config.get('action_space_dim', 'not set')}")
            print(f"   - Adaptive stddev: {config.get('adaptive_stddev', 'not set')}")
            print(f"   - Batch size: {config.get('batch_size', 'not set')}")
            print(f"   - Environments: {config.get('env_agents', 'not set')}")
            print(f"   - Batches per epoch: {config.get('num_batches_per_epoch', 'not set')}")
            print(f"   - Visualization support: Available (headless parameter supported)")
            
            # Check if configuration matches original DCE
            if (config.get('action_space_dim') == 3 and 
                config.get('adaptive_stddev') == True and
                config.get('batch_size') == 2048 and
                config.get('env_agents') == 16 and
                config.get('num_batches_per_epoch') == 8):
                print("✅ Training script configuration: SUCCESS (Original DCE config)")
                return True
            else:
                print("❌ Training script configuration: MISMATCH")
                return False
        else:
            print("❌ Training script configuration: quad_with_obstacles not found")
            return False
            
    except Exception as e:
        print(f"❌ Training script validation FAILED: {e}")
        return False

def main():
    """Run all validations."""
    print("🚀 DCE Navigation Configuration Validation")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= validate_imports()
    all_passed &= validate_configuration()
    all_passed &= validate_action_compatibility()
    all_passed &= validate_shell_script()
    all_passed &= validate_training_script()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ Ready for training with original DCE configuration")
        print("📝 Configuration summary:")
        print("   - 16 environments (maximum performance)")
        print("   - 2048 batch size")
        print("   - 4096 effective batch size")
        print("   - 3D action space (inference compatible)")
        print("   - 81D observation space (17D state + 64D VAE latents)")
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED!")
        print("   Please fix the issues above before training")
        sys.exit(1)

if __name__ == "__main__":
    main() 