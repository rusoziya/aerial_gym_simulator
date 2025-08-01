"""
Debug script to identify gate scaling issues.

This script tests the gate scaling functionality to find why
medium gates aren't being applied at curriculum level 9.
"""

import sys
import os

# Add the aerial_gym package to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import IsaacGym first, then PyTorch
try:
    import isaacgym
    import torch
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure IsaacGym is properly installed and in your PYTHONPATH")
    sys.exit(1)

def test_gate_scaling_at_level_9():
    """Test gate scaling functionality at curriculum level 9."""
    
    print("🔍 DEBUGGING GATE SCALING AT LEVEL 9")
    print("=" * 60)
    
    try:
        # Test 1: Import and basic functionality
        print("1. Testing imports...")
        from aerial_gym.config.asset_config.gate_scaling_config import GateScalingConfig
        from aerial_gym.config.task_config.navigation_task_config_gate import task_config
        print("   ✅ Imports successful")
        
        # Test 2: Check available scales for level 9
        print("\n2. Testing available scales...")
        level_9_scales = GateScalingConfig.get_available_scales_for_level(9)
        print(f"   Level 9 available scales: {level_9_scales}")
        
        if 0.7 in level_9_scales:
            print("   ✅ Medium gates (0.7) should be available at level 9")
        else:
            print("   ❌ Medium gates (0.7) NOT available at level 9 - THIS IS THE ISSUE!")
        
        # Test 3: Check scale selection
        print("\n3. Testing scale selection...")
        for i in range(10):
            selected_scale = task_config.curriculum.get_gate_scale_for_level(9)
            print(f"   Selection {i+1}: {selected_scale}")
        
        # Test 4: Check tolerance calculation
        print("\n4. Testing tolerance calculation...")
        for scale in [1.0, 0.7]:
            if scale in level_9_scales:
                width_tol, height_min, height_max = GateScalingConfig.get_gate_tolerance_for_scale(scale)
                print(f"   Scale {scale}: width ±{width_tol:.2f}m, height {height_min:.1f}-{height_max:.1f}m")
        
        # Test 5: Simulate the issue scenario
        print("\n5. Simulating curriculum advancement...")
        print("   Previous level 8 scales:", GateScalingConfig.get_available_scales_for_level(8))
        print("   New level 9 scales:", GateScalingConfig.get_available_scales_for_level(9))
        print("   Should see medium gates (0.7) added at level 9")
        
        # Test 6: Check potential issues
        print("\n6. Checking for potential issues...")
        
        # Issue A: Import path problems
        try:
            import aerial_gym.config.asset_config.gate_scaling_config
            print("   ✅ Gate scaling config module loads correctly")
        except Exception as e:
            print(f"   ❌ Gate scaling config import error: {e}")
        
        # Issue B: Method availability
        if hasattr(GateScalingConfig, 'get_available_scales_for_level'):
            print("   ✅ get_available_scales_for_level method exists")
        else:
            print("   ❌ get_available_scales_for_level method missing")
        
        if hasattr(task_config.curriculum, 'get_gate_scale_for_level'):
            print("   ✅ get_gate_scale_for_level method exists")
        else:
            print("   ❌ get_gate_scale_for_level method missing")
        
        # Issue C: Logic flow test
        print("\n7. Testing logic flow...")
        test_levels = [8, 9, 10]
        for level in test_levels:
            available = GateScalingConfig.get_available_scales_for_level(level)
            selected = task_config.curriculum.get_gate_scale_for_level(level)
            print(f"   Level {level}: available={available}, selected={selected}")
            
            if selected not in available:
                print(f"   ❌ LOGIC ERROR: Selected scale {selected} not in available scales {available}")
            else:
                print(f"   ✅ Logic correct for level {level}")
        
        print("\n" + "=" * 60)
        print("✅ Gate scaling test completed!")
        
        # Summary
        print("\n📋 SUMMARY:")
        print(f"   - Level 9 should have scales: {level_9_scales}")
        print("   - Medium gates (0.7) should be available")
        print("   - Gate scaling only applies on environment RESET")
        print("   - Check the debug logs during training to see if:")
        print("     1. Scale selection is working")
        print("     2. Environments are actually resetting")
        print("     3. Tolerances are being updated")
        
    except Exception as e:
        print(f"❌ DEBUG TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

def test_curriculum_progression():
    """Test the full curriculum progression."""
    
    print("\n🎯 CURRICULUM PROGRESSION TEST")
    print("=" * 60)
    
    try:
        from aerial_gym.config.asset_config.gate_scaling_config import GateScalingConfig
        
        level_ranges = [
            (3, 8, "Full Only"),
            (9, 13, "Full + Medium"), 
            (14, 18, "Full + Med + Small"),
            (19, 23, "All Gates")
        ]
        
        print(f"{'Level Range':<12} {'Description':<20} {'Available Scales'}")
        print("-" * 55)
        
        for start, end, desc in level_ranges:
            sample_level = (start + end) // 2
            scales = GateScalingConfig.get_available_scales_for_level(sample_level)
            print(f"{start}-{end:<9} {desc:<20} {scales}")
        
        # Focus on the problematic transition
        print(f"\n🔍 LEVEL 8 → 9 TRANSITION:")
        level_8_scales = GateScalingConfig.get_available_scales_for_level(8)
        level_9_scales = GateScalingConfig.get_available_scales_for_level(9)
        
        print(f"   Level 8: {level_8_scales}")
        print(f"   Level 9: {level_9_scales}")
        
        if len(level_9_scales) > len(level_8_scales):
            print("   ✅ More scales available at level 9 (expected)")
            added_scales = set(level_9_scales) - set(level_8_scales)
            print(f"   New scales added: {list(added_scales)}")
        else:
            print("   ❌ No new scales added at level 9 (unexpected)")
        
    except Exception as e:
        print(f"❌ Curriculum progression test failed: {e}")

def identify_reset_timing_issue():
    """Identify if the issue is related to reset timing."""
    
    print("\n⏰ RESET TIMING ANALYSIS")
    print("=" * 60)
    
    print("POTENTIAL ISSUE: Gate scaling only applies on environment RESET")
    print("")
    print("Timeline of events:")
    print("1. Curriculum level increases from 8 → 9")
    print("2. New scales become available: [1.0, 0.7]")
    print("3. ⚠️  BUT existing environments keep current gate scale (1.0)")
    print("4. Gate scaling only updates when environments RESET")
    print("")
    print("Possible causes:")
    print("- Environments not resetting frequently enough")
    print("- Reset not triggering gate scaling logic")
    print("- Asset positioning not working")
    print("- Scale selection returning wrong values")
    print("")
    print("Solutions to check:")
    print("1. Force some environments to reset after curriculum change")
    print("2. Verify reset_idx is being called")
    print("3. Check if _apply_curriculum_gate_scaling is being called")
    print("4. Verify asset manager integration")

if __name__ == "__main__":
    test_gate_scaling_at_level_9()
    test_curriculum_progression()
    identify_reset_timing_issue()
    
    print("\n🚀 NEXT STEPS:")
    print("1. Run your training and watch for the new debug logs")
    print("2. Look for '[GATE SCALING RESET]' messages when envs reset")
    print("3. Check '[GATE SCALE SELECT DEBUG]' for scale selection")
    print("4. Monitor '[TOLERANCE DEBUG]' for actual tolerance values")
    print("5. Verify that some environments reset after curriculum advances") 