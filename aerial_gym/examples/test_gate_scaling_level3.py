"""
Test script to verify gate scaling functionality at curriculum level 3.

This script tests the gate scaling configuration to ensure it works
correctly at early curriculum levels.
"""

import sys
import os

# Add the aerial_gym package to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import IsaacGym first, then other modules
try:
    import isaacgym
    import torch
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure IsaacGym is properly installed and in your PYTHONPATH")
    sys.exit(1)

def test_level_3_gate_scaling():
    """Test gate scaling configuration at curriculum level 3."""
    
    print("🔍 TESTING GATE SCALING AT LEVEL 3")
    print("=" * 60)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from aerial_gym.config.asset_config.gate_scaling_config import GateScalingConfig
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        print("   ✅ Imports successful")
        
        # Test available scales at level 3
        print("\n2. Testing available scales at level 3...")
        level_3_scales = GateScalingConfig.get_available_scales_for_level(3)
        print(f"   Available scales: {level_3_scales}")
        
        expected_scales = [1.0]  # Only full-size gates at level 3
        if level_3_scales == expected_scales:
            print("   ✅ Correct: Only full-size gates (1.0) available at level 3")
        else:
            print(f"   ❌ Unexpected scales. Expected {expected_scales}, got {level_3_scales}")
        
        # Test scale selection multiple times
        print("\n3. Testing scale selection (should always be 1.0)...")
        for i in range(5):
            selected_scale = task_config.curriculum.get_gate_scale_for_level(3)
            print(f"   Selection {i+1}: {selected_scale}")
            if selected_scale != 1.0:
                print(f"   ❌ Unexpected scale {selected_scale}, expected 1.0")
                return False
        
        print("   ✅ All selections correctly returned 1.0")
        
        # Test tolerance calculation for scale 1.0
        print("\n4. Testing tolerance calculation for full-size gate...")
        width_tol, height_min, height_max = GateScalingConfig.get_gate_tolerance_for_scale(1.0)
        print(f"   Scale 1.0 tolerance: ±{width_tol:.2f}m width, {height_min:.1f}-{height_max:.1f}m height")
        
        expected_width = 1.3  # Base width tolerance
        expected_height_min = 0.2  # Base height min
        expected_height_max = 2.2  # Base height max
        
        if (abs(width_tol - expected_width) < 0.01 and 
            abs(height_min - expected_height_min) < 0.01 and 
            abs(height_max - expected_height_max) < 0.01):
            print("   ✅ Tolerance calculation correct for full-size gate")
        else:
            print(f"   ❌ Tolerance mismatch. Expected ±{expected_width}m, {expected_height_min}-{expected_height_max}m")
            print(f"      Got ±{width_tol:.2f}m, {height_min:.1f}-{height_max:.1f}m")
        
        # Test curriculum progression expectations
        print("\n5. Testing curriculum progression...")
        progression_tests = [
            (3, [1.0], "Level 3: Full only"),
            (8, [1.0], "Level 8: Full only"), 
            (9, [1.0, 0.7], "Level 9: Full + Medium"),
            (13, [1.0, 0.7], "Level 13: Full + Medium"),
            (14, [1.0, 0.7, 0.5], "Level 14: Full + Med + Small"),
            (19, [1.0, 0.7, 0.5, 0.4], "Level 19: All gates")
        ]
        
        for level, expected, description in progression_tests:
            actual = GateScalingConfig.get_available_scales_for_level(level)
            status = "✅" if actual == expected else "❌"
            print(f"   {status} {description}: {actual}")
        
        print("\n" + "=" * 60)
        print("✅ Level 3 gate scaling test completed!")
        
        print("\n📋 SUMMARY FOR LEVEL 3:")
        print("   - Should only use full-size gates (scale 1.0)")
        print("   - Tolerance: ±1.3m width, 0.2-2.2m height")
        print("   - No gate size variation expected")
        print("   - Debug logs should show 'full' gate type selection")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_debug_logging_setup():
    """Check if debug logging is properly configured."""
    
    print("\n🔧 DEBUG LOGGING CHECK")
    print("=" * 60)
    
    try:
        from aerial_gym.utils.logging import CustomLogger
        
        # Create a test logger
        test_logger = CustomLogger("test_gate_scaling")
        
        print("Available logging levels:")
        print("   - WARNING: Should always be visible")
        print("   - DEBUG: May be filtered depending on log level")
        
        print("\nTest logging output:")
        test_logger.warning("   ✅ WARNING level message (should be visible)")
        test_logger.debug("   🔍 DEBUG level message (may not be visible)")
        
        print("\nIf you don't see WARNING messages during training:")
        print("   1. Check your logging configuration")
        print("   2. Look for log files in aerial_gym/logs/")
        print("   3. Ensure logger level is set to WARNING or lower")
        
    except Exception as e:
        print(f"❌ Logging check failed: {e}")

if __name__ == "__main__":
    success = test_level_3_gate_scaling()
    check_debug_logging_setup()
    
    if success:
        print("\n🚀 NEXT STEPS:")
        print("1. Run training and look for '[GATE SCALING INIT]' messages")
        print("2. Check for '[GATE SCALING RESET]' when environments reset")
        print("3. Look for '[GATE SCALING] Env X: Selected scale 1.0' messages")
        print("4. At level 3, all selections should be scale 1.0 (full gates)")
        print("5. At level 9+, you should see scale 0.7 selections")
    else:
        print("\n❌ Gate scaling configuration has issues - fix before testing training") 