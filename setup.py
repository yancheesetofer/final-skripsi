#!/usr/bin/env python3
"""
Setup script for Moodle Cheating Detection System
Automates installation and environment preparation
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n📋 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 8 <= version.minor <= 10:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python version not supported")
        print("Required: Python 3.8, 3.9, or 3.10")
        print("Please install a compatible Python version")
        return False

def create_directories():
    """Create required directory structure"""
    print_header("Creating Directory Structure")
    
    directories = [
        "data",
        "data/synthetic_moodle_logs",
        "data/real_moodle_logs",
        "results",
        "detection_results",
        "preprocessing/artifacts",
        "draft-skripsi-tex/newfigures",
        "logs"
    ]
    
    success = True
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created/verified: {directory}")
        except Exception as e:
            print(f"❌ Failed to create {directory}: {e}")
            success = False
    
    return success

def setup_virtual_environment():
    """Create and activate virtual environment"""
    print_header("Setting up Virtual Environment")
    
    # Check if already in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("📝 Already in a virtual environment")
        return True
    
    # Create virtual environment
    venv_name = "venv"
    if not os.path.exists(venv_name):
        success = run_command(f"{sys.executable} -m venv {venv_name}", 
                            "Creating virtual environment")
        if not success:
            return False
    else:
        print(f"✅ Virtual environment '{venv_name}' already exists")
    
    # Provide activation instructions
    system = platform.system().lower()
    if system == "windows":
        activate_cmd = f"{venv_name}\\Scripts\\activate"
    else:
        activate_cmd = f"source {venv_name}/bin/activate"
    
    print(f"\n📝 To activate the virtual environment, run:")
    print(f"   {activate_cmd}")
    print(f"\n📝 After activation, run this script again to continue setup")
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    print_header("Installing Dependencies")
    
    # Upgrade pip first
    success = run_command(f"{sys.executable} -m pip install --upgrade pip", 
                         "Upgrading pip")
    if not success:
        return False
    
    # Install requirements
    success = run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                         "Installing requirements")
    if not success:
        return False
    
    # Verify critical imports
    print("\n📋 Verifying critical package installations...")
    critical_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 
        'matplotlib', 'seaborn', 'joblib'
    ]
    
    for package in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Import failed")
            return False
    
    return True

def create_test_config():
    """Create a test configuration file"""
    print_header("Creating Test Configuration")
    
    test_config = {
        "num_users": 100,  # Smaller for testing
        "num_quizzes": 10,
        "base_date": "2024-01-01",
        "timelimit": 3600,
        "cheating_groups": [
            {
                "id": "test_group_1",
                "severity": "high",
                "members": [1, 2, 3, 4, 5],
                "patterns": {
                    "navigation_similarity": 0.95,
                    "timing_correlation": 0.9,
                    "answer_similarity": 0.9,
                    "wrong_answer_bias": 0.8
                }
            }
        ]
    }
    
    import json
    try:
        with open("test_config.json", "w") as f:
            json.dump(test_config, f, indent=2)
        print("✅ Created test_config.json for testing")
        return True
    except Exception as e:
        print(f"❌ Failed to create test config: {e}")
        return False

def run_validation():
    """Run basic validation of the setup"""
    print_header("Running Setup Validation")
    
    validation_script = """
import sys
import os
import pandas as pd
import numpy as np
import sklearn
import xgboost
import matplotlib
import seaborn
from pathlib import Path

print("Python executable:", sys.executable)
print("Python version:", sys.version)

# Check directories
required_dirs = ['data', 'results', 'detection_results', 'preprocessing/artifacts']
for dir_name in required_dirs:
    if Path(dir_name).exists():
        print(f"✅ Directory exists: {dir_name}")
    else:
        print(f"❌ Directory missing: {dir_name}")

# Check key files
key_files = ['generate_case.py', 'enhanced_model.py', 'cheating_detection.py', 'requirements.txt']
for file_name in key_files:
    if Path(file_name).exists():
        print(f"✅ File exists: {file_name}")
    else:
        print(f"❌ File missing: {file_name}")

print("✅ Validation complete!")
"""
    
    try:
        with open("validate_setup.py", "w") as f:
            f.write(validation_script)
        
        success = run_command(f"{sys.executable} validate_setup.py", 
                            "Running validation script")
        
        # Clean up
        if os.path.exists("validate_setup.py"):
            os.remove("validate_setup.py")
        
        return success
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    """Main setup function"""
    print_header("Moodle Cheating Detection System Setup")
    print("This script will set up your environment for the cheating detection system")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if not in_venv:
        print("\n⚠️  Not in virtual environment. Setting up virtual environment...")
        if setup_virtual_environment():
            print("\n📝 Please activate the virtual environment and run this script again:")
            if platform.system().lower() == "windows":
                print("   venv\\Scripts\\activate")
            else:
                print("   source venv/bin/activate")
            print("   python setup.py")
        sys.exit(0)
    
    print("✅ Running in virtual environment")
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create test configuration
    if not create_test_config():
        print("❌ Failed to create test configuration")
        sys.exit(1)
    
    # Run validation
    if not run_validation():
        print("❌ Setup validation failed")
        sys.exit(1)
    
    # Success message
    print_header("Setup Complete!")
    print("✅ Environment setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run: python generate_case.py")
    print("2. Run: python -m preprocessing.main_preprocessor --data_input_dir data/synthetic_moodle_logs --output_feature_path data/processed_artificial_features_V2.csv --train_mode --ground_truth_path data/synthetic_ground_truth.csv")
    print("3. Run: python enhanced_model.py")
    print("\n📖 For detailed instructions, see README.md")

if __name__ == "__main__":
    main() 