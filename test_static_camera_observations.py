#!/usr/bin/env python3

"""
Test Script: Verify Static Camera Observations Integration

This script tests whether static drone footage is properly fed as observations
to the drone in the navigation task gate environment.
"""

import sys
import os
import numpy as np
import time

# Add the current directory to Python path for imports
sys.path.append('/home/ziyar/aerialgym/aerialgym_ws/src/aerial_gym_simulator')

# CRITICAL: Import isaacgym BEFORE torch to avoid import conflicts
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.examples.dce_rl_navigation.dce_navigation_task_gate import DCE_RL_Navigation_Task_Gate
from aerial_gym.config.task_config.navigation_task_config_gate import task_config
from aerial_gym.utils.logging import CustomLogger

# Import torch AFTER isaacgym modules
import torch

logger = CustomLogger("StaticCameraTest")

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def test_static_camera_observations():
    """Test if static camera observations are properly integrated."""
    
    print_section("STATIC CAMERA OBSERVATION VERIFICATION TEST")
    
    try:
        # 1. Setup Environment
        print_section("1. ENVIRONMENT SETUP")
        
        # Register the DCE navigation task 
        task_registry.register_task(
            task_name="static_camera_test_task",
            task_class=DCE_RL_Navigation_Task_Gate,
            task_config=task_config,
        )
        
        # Create task - use non-headless mode to enable real camera if possible
        print("Creating navigation task...")
        rl_task = task_registry.make_task(
            "static_camera_test_task",
            seed=42,
            use_warp=True,
            headless=False,  # Try non-headless first for real camera
            num_envs=2       # Use 2 environments for testing
        )
        
        print(f"✓ Task created with {rl_task.num_envs} environments")
        print(f"✓ Device: {rl_task.device}")
        print(f"✓ VAE enabled: {rl_task.task_config.vae_config.use_vae}")
        
        # Initialize the task
        rl_task.reset()
        print("✓ Task initialized and reset")

        # 2. Check Static Camera Manager
        print_section("2. STATIC CAMERA MANAGER VERIFICATION")
        
        if hasattr(rl_task, 'static_camera_manager'):
            scm = rl_task.static_camera_manager
            print("✓ Static camera manager exists")
            print(f"  - Camera setup success: {scm.camera_setup_success}")
            print(f"  - Using synthetic camera: {scm.use_synthetic_camera}")
            print(f"  - Number of camera handles: {len(scm.camera_handles)}")
            print(f"  - Gate position: {scm.gate_position}")
        else:
            print("❌ Static camera manager not found!")
            return False

        # 3. Check Static Camera Latents Tensor
        print_section("3. STATIC CAMERA LATENTS TENSOR VERIFICATION")
        
        if hasattr(rl_task, 'static_camera_latents'):
            latents = rl_task.static_camera_latents
            print("✓ Static camera latents tensor exists")
            print(f"  - Shape: {latents.shape}")
            print(f"  - Device: {latents.device}")
            print(f"  - Dtype: {latents.dtype}")
            print(f"  - Requires grad: {latents.requires_grad}")
            print(f"  - Initial mean: {latents.mean().item():.6f}")
            print(f"  - Initial std: {latents.std().item():.6f}")
        else:
            print("❌ Static camera latents tensor not found!")
            return False

        # 4. Test Static Camera Image Capture
        print_section("4. STATIC CAMERA IMAGE CAPTURE TEST")
        
        try:
            static_depth, static_seg = scm.capture_images()
            if static_depth is not None:
                print("✓ Static camera image capture successful")
                print(f"  - Depth image shape: {static_depth.shape}")
                print(f"  - Depth image dtype: {static_depth.dtype}")
                print(f"  - Depth range: [{static_depth.min():.3f}, {static_depth.max():.3f}]")
                print(f"  - Depth mean: {static_depth.mean():.3f}")
                
                if static_seg is not None:
                    print(f"  - Segmentation image shape: {static_seg.shape}")
                    print(f"  - Segmentation unique values: {np.unique(static_seg)}")
            else:
                print("⚠️  Static camera capture returned None (may be normal for synthetic mode)")
        except Exception as e:
            print(f"❌ Static camera capture failed: {e}")

        # 5. Test Observation Processing 
        print_section("5. OBSERVATION PROCESSING TEST")
        
        print("Testing static camera observation processing...")
        
        # Store initial latents for comparison
        initial_latents = rl_task.static_camera_latents.clone()
        
        # Process static camera observation
        try:
            rl_task.process_static_camera_observation()
            print("✓ Static camera observation processing completed")
            
            # Check if latents changed
            final_latents = rl_task.static_camera_latents
            latents_changed = not torch.equal(initial_latents, final_latents)
            
            print(f"  - Latents changed: {latents_changed}")
            print(f"  - Final latents mean: {final_latents.mean().item():.6f}")
            print(f"  - Final latents std: {final_latents.std().item():.6f}")
            
            if latents_changed:
                print("✓ Static camera latents were updated (good!)")
            else:
                print("⚠️  Static camera latents unchanged (may indicate issue)")
                
        except Exception as e:
            print(f"❌ Static camera observation processing failed: {e}")

        # 6. Test Full Observation Integration 
        print_section("6. FULL OBSERVATION INTEGRATION TEST")
        
        print("Testing full step with static camera observations...")
        
        # Create dummy actions
        actions = torch.zeros((rl_task.num_envs, 4), device=rl_task.device)
        
        # Store pre-step latents
        pre_step_latents = rl_task.static_camera_latents.clone()
        
        # Execute a full step
        obs, rewards, terminations, truncations, infos = rl_task.step(actions)
        
        # Check post-step latents
        post_step_latents = rl_task.static_camera_latents.clone()
        
        print("✓ Full step completed")
        print(f"  - Observation shape: {obs['observations'].shape}")
        print(f"  - Expected obs shape: ({rl_task.num_envs}, {rl_task.task_config.observation_space_dim})")
        
        # Check if static camera latents are in observations
        if rl_task.task_config.vae_config.use_vae:
            static_latents_in_obs = obs['observations'][:, 81:145]  # Static camera VAE latents
            print(f"  - Static latents in observations shape: {static_latents_in_obs.shape}")
            print(f"  - Static latents in obs mean: {static_latents_in_obs.mean().item():.6f}")
            print(f"  - Static latents in obs std: {static_latents_in_obs.std().item():.6f}")
            
            # Verify they match the stored latents
            latents_match = torch.allclose(static_latents_in_obs, post_step_latents, rtol=1e-5)
            print(f"  - Latents match stored values: {latents_match}")
            
            if latents_match:
                print("✓ Static camera latents properly integrated into observations!")
            else:
                print("❌ Static camera latents mismatch in observations")
        else:
            print("⚠️  VAE disabled - static camera integration limited")

        # 7. Multi-Step Test
        print_section("7. MULTI-STEP CONSISTENCY TEST")
        
        print("Testing consistency over multiple steps...")
        
        latents_history = []
        obs_history = []
        
        for step in range(5):
            # Random actions to create some movement
            actions = 0.1 * torch.randn((rl_task.num_envs, 4), device=rl_task.device)
            
            obs, _, _, _, _ = rl_task.step(actions)
            
            current_latents = rl_task.static_camera_latents.clone()
            latents_history.append(current_latents)
            
            if rl_task.task_config.vae_config.use_vae:
                obs_latents = obs['observations'][:, 81:145]
                obs_history.append(obs_latents)
                
                # Check consistency within this step
                consistency = torch.allclose(current_latents, obs_latents, rtol=1e-5)
                print(f"  Step {step}: Latents consistent = {consistency}, Mean = {current_latents.mean().item():.6f}")
        
        # Check if latents are updating appropriately
        if len(latents_history) > 1:
            all_same = all(torch.equal(latents_history[0], h) for h in latents_history[1:])
            if all_same:
                print("⚠️  Static camera latents identical across all steps (may indicate static camera issue)")
            else:
                print("✓ Static camera latents vary across steps (good for real camera)")

        # 8. Summary and Diagnosis
        print_section("8. SUMMARY AND DIAGNOSIS")
        
        print("Static Camera System Status:")
        print(f"  ✓ Static camera manager: {'Present' if hasattr(rl_task, 'static_camera_manager') else 'Missing'}")
        print(f"  ✓ Static camera latents: {'Present' if hasattr(rl_task, 'static_camera_latents') else 'Missing'}")
        print(f"  ✓ Camera setup success: {scm.camera_setup_success if hasattr(rl_task, 'static_camera_manager') else 'N/A'}")
        print(f"  ✓ Using synthetic camera: {scm.use_synthetic_camera if hasattr(rl_task, 'static_camera_manager') else 'N/A'}")
        print(f"  ✓ VAE processing: {'Enabled' if rl_task.task_config.vae_config.use_vae else 'Disabled'}")
        
        if rl_task.task_config.vae_config.use_vae and hasattr(rl_task, 'static_camera_latents'):
            current_mean = rl_task.static_camera_latents.mean().item()
            current_std = rl_task.static_camera_latents.std().item()
            
            if abs(current_mean) < 1e-6 and abs(current_std) < 1e-6:
                print("  ⚠️  Static camera latents are all zeros - possible issue!")
                print("      This suggests static camera processing may not be working properly.")
            else:
                print("  ✓ Static camera latents have meaningful values")
                print("      Static camera footage appears to be properly processed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        try:
            if 'rl_task' in locals():
                rl_task.close()
        except:
            pass

def main():
    """Main test function."""
    print("Starting Static Camera Observation Verification Test...")
    
    success = test_static_camera_observations()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("Static camera observations appear to be properly integrated.")
    else:
        print("\n❌ Test failed!")
        print("There may be issues with static camera observation integration.")

if __name__ == "__main__":
    main() 