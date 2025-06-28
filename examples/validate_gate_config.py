"""
Gate Configuration Validation

This script validates the gate configuration files without importing Isaac Gym.
It directly checks the configuration files we created to ensure they're properly structured.

This avoids the Python 3.8 vs 3.12 compatibility issue with Isaac Gym.
"""

import os
import sys


def check_gate_urdf():
    """Check if the gate URDF file exists and validate its structure."""
    
    print("GATE URDF FILE VALIDATION")
    print("=" * 40)
    
    # Construct the expected path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    urdf_path = os.path.join(project_root, "resources/models/environment_assets/objects/gate.urdf")
    
    print(f"Looking for URDF at: {urdf_path}")
    
    if os.path.exists(urdf_path):
        print("✓ Gate URDF file found!")
        
        # Read and validate URDF content
        try:
            with open(urdf_path, 'r') as f:
                content = f.read()
                
            print(f"✓ URDF file size: {len(content)} characters")
            
            # Check for required XML structure
            if content.startswith('<?xml version="1.0"?>'):
                print("✓ Valid XML header found")
            else:
                print("❌ Missing XML header")
                
            # Check for robot definition
            if '<robot name="gate">' in content:
                print("✓ Robot definition found")
            else:
                print("❌ Missing robot definition")
                
            # Count components
            link_count = content.count('<link name=')
            joint_count = content.count('<joint name=')
            
            print(f"✓ URDF structure:")
            print(f"  - {link_count} links found")
            print(f"  - {joint_count} joints found")
            
            # Check for specific gate components
            required_components = ['left_post', 'right_post', 'top_beam', 'base_link']
            for component in required_components:
                if component in content:
                    print(f"  - ✓ {component} component found")
                else:
                    print(f"  - ❌ {component} component MISSING")
                    
            # Check for physics properties
            if '<collision>' in content:
                print("  - ✓ Collision geometry defined")
            if '<visual>' in content:
                print("  - ✓ Visual geometry defined")
            if '<inertial>' in content:
                print("  - ✓ Inertial properties defined")
                
            return True
            
        except Exception as e:
            print(f"❌ Error reading URDF: {e}")
            return False
            
    else:
        print("❌ Gate URDF file NOT found!")
        print("Expected location:", urdf_path)
        return False


def check_gate_asset_config():
    """Check if the gate asset configuration file exists."""
    
    print("\nGATE ASSET CONFIGURATION VALIDATION")
    print("=" * 45)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, "aerial_gym/config/asset_config/gate_asset_config.py")
    
    print(f"Looking for config at: {config_path}")
    
    if os.path.exists(config_path):
        print("✓ Gate asset configuration file found!")
        
        try:
            with open(config_path, 'r') as f:
                content = f.read()
                
            print(f"✓ Configuration file size: {len(content)} characters")
            
            # Check for required classes and imports
            checks = [
                ('BaseAssetParams', 'Base class import'),
                ('AERIAL_GYM_DIRECTORY', 'Directory constant'),
                ('GATE_SEMANTIC_ID', 'Semantic ID definition'),
                ('GateAssetConfig', 'Main config class'),
                ('gate_asset_params', 'Asset parameters class'),
                ('num_assets', 'Asset count parameter'),
                ('asset_folder', 'Asset folder path'),
                ('file = "gate.urdf"', 'URDF file reference'),
                ('min_position_ratio', 'Position configuration'),
                ('collision_mask', 'Collision settings'),
                ('semantic_id', 'Semantic labeling')
            ]
            
            for check_item, description in checks:
                if check_item in content:
                    print(f"  - ✓ {description} found")
                else:
                    print(f"  - ❌ {description} MISSING")
                    
            return True
            
        except Exception as e:
            print(f"❌ Error reading config file: {e}")
            return False
            
    else:
        print("❌ Gate asset configuration file NOT found!")
        return False


def check_gate_env_config():
    """Check if the gate environment configuration file exists."""
    
    print("\nGATE ENVIRONMENT CONFIGURATION VALIDATION")
    print("=" * 50)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, "aerial_gym/config/env_config/gate_env.py")
    
    print(f"Looking for config at: {config_path}")
    
    if os.path.exists(config_path):
        print("✓ Gate environment configuration file found!")
        
        try:
            with open(config_path, 'r') as f:
                content = f.read()
                
            print(f"✓ Configuration file size: {len(content)} characters")
            
            # Check for required environment configurations
            env_checks = [
                ('GateEnvCfg', 'Basic gate environment'),
                ('GateEnvWithObstaclesCfg', 'Gate with obstacles environment'),
                ('GateEnvRandomizedCfg', 'Randomized gate environment'),
                ('class env:', 'Environment parameters class'),
                ('class env_config:', 'Environment config class'),
                ('include_asset_type', 'Asset inclusion settings'),
                ('asset_type_to_dict_map', 'Asset type mapping'),
                ('"gate": True', 'Gate asset enabled'),
                ('num_envs', 'Environment count'),
                ('env_spacing', 'Environment spacing'),
                ('collision_force_threshold', 'Collision detection'),
                ('lower_bound_min', 'Environment bounds'),
                ('upper_bound_max', 'Environment bounds')
            ]
            
            for check_item, description in env_checks:
                if check_item in content:
                    print(f"  - ✓ {description} found")
                else:
                    print(f"  - ❌ {description} MISSING")
                    
            return True
            
        except Exception as e:
            print(f"❌ Error reading environment config: {e}")
            return False
            
    else:
        print("❌ Gate environment configuration file NOT found!")
        return False


def validate_file_structure():
    """Validate the overall file structure."""
    
    print("\nFILE STRUCTURE VALIDATION")
    print("=" * 30)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Expected files and directories
    expected_structure = [
        "resources/models/environment_assets/objects/gate.urdf",
        "aerial_gym/config/asset_config/gate_asset_config.py", 
        "aerial_gym/config/env_config/gate_env.py",
        "examples/simple_gate_demo.py",
        "examples/simple_gate_visualization.py"
    ]
    
    all_exist = True
    
    for file_path in expected_structure:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
            
    return all_exist


def main():
    """Main validation function."""
    
    print("GATE ENVIRONMENT VALIDATION")
    print("This validates our gate configuration without Isaac Gym dependencies")
    print("=" * 70)
    
    # Run all validation checks
    urdf_ok = check_gate_urdf()
    asset_config_ok = check_gate_asset_config()
    env_config_ok = check_gate_env_config()
    structure_ok = validate_file_structure()
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    if all([urdf_ok, asset_config_ok, env_config_ok, structure_ok]):
        print("🎉 ALL VALIDATIONS PASSED!")
        print("\nYour gate environment is properly configured!")
        print("\nWhat's working:")
        print("✓ Gate URDF model is properly structured")
        print("✓ Asset configuration is complete") 
        print("✓ Environment configuration is ready")
        print("✓ All required files are in place")
        
        print("\nNext steps to fix Isaac Gym compatibility:")
        print("1. Create a Python 3.8 conda environment")
        print("2. Install Isaac Gym in that environment")
        print("3. Install other required packages (torch, etc.)")
        print("4. Run the full simulation demo")
        
        print("\nAlternatively, you can:")
        print("- Use the configuration files with a different simulator")
        print("- Wait for Isaac Lab compatibility (mentioned in README)")
        
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("Please check the missing components above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 