#!/usr/bin/env python3
"""
Quick Start Script for Moodle Cheating Detection System
Runs the complete pipeline with default settings for new users

This script is perfect for:
- First-time users who want to see the system in action
- Supervisors who want to quickly validate the research
- Testing the system in a new environment

Usage: python quick_start.py [--small] [--no-plots]
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("="*70)
    print("🎓 MOODLE CHEATING DETECTION SYSTEM - QUICK START")
    print("   PEMANTAUAN KEPATUHAN SECARA OTOMATIS")
    print("   MELALUI ANALISIS LOG PADA MOODLE BERBASIS KECERDASAN BUATAN")
    print("="*70)

def print_step(step_num, title, description):
    """Print step header"""
    print(f"\n{'='*50}")
    print(f"📋 STEP {step_num}: {title}")
    print(f"   {description}")
    print("="*50)

def run_command_with_progress(command, description):
    """Run command with progress indication"""
    print(f"\n🔄 {description}...")
    print(f"💻 Command: {command}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        elapsed = time.time() - start_time
        print(f"✅ {description} completed successfully in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        return False

def check_environment():
    """Check if environment is ready"""
    print_step(0, "ENVIRONMENT CHECK", "Verifying system requirements")
    
    # Check Python version
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if not (version.major == 3 and 8 <= version.minor <= 10):
        print("❌ Python version not supported. Need Python 3.8-3.10")
        return False
    
    # Check key files
    required_files = [
        'generate_case.py',
        'enhanced_model.py', 
        'cheating_detection.py',
        'requirements.txt'
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("⚠️  Not in virtual environment. Run setup.py first!")
        return False
    
    # Try importing key packages
    try:
        import pandas, numpy, sklearn, xgboost
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ Environment ready!")
    return True

def create_directories():
    """Create necessary directories"""
    dirs = ['data', 'results', 'detection_results', 'preprocessing/artifacts']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def quick_start_pipeline(small_dataset=False, no_plots=False):
    """Run the complete pipeline"""
    
    print_banner()
    
    if not check_environment():
        print("\n❌ Environment check failed. Please run setup.py first.")
        return False
    
    create_directories()
    
    # Step 1: Generate Data
    print_step(1, "DATA GENERATION", 
               f"Creating {'small' if small_dataset else 'standard'} synthetic dataset")
    
    config_file = "synthetic_data_config_800.json" if not small_dataset else None
    cmd = f"python generate_case.py {config_file}" if config_file else "python generate_case.py"
    
    if not run_command_with_progress(cmd, "Generate synthetic Moodle data"):
        return False
    
    # Step 2: Preprocessing
    print_step(2, "DATA PREPROCESSING", "Extracting features and preparing data for ML")
    
    cmd = ("python -m preprocessing.main_preprocessor "
           "--data_input_dir data/synthetic_moodle_logs "
           "--output_feature_path data/processed_artificial_features_V2.csv "
           "--train_mode "
           "--ground_truth_path data/synthetic_ground_truth.csv")
    
    if not run_command_with_progress(cmd, "Process and extract features"):
        return False
    
    # Step 3: Model Training
    print_step(3, "MODEL TRAINING", "Training ensemble of ML algorithms")
    
    if not run_command_with_progress("python enhanced_model.py", "Train ML models"):
        return False
    
    # Step 4: Detection Demo
    print_step(4, "DETECTION DEMO", "Applying models to detect cheating patterns")
    
    # Copy synthetic features to simulate real data detection
    import shutil
    shutil.copy("data/processed_artificial_features_V2.csv",
                "data/processed_real_features_for_detection_V2.csv")
    
    if not run_command_with_progress("python cheating_detection.py", "Run cheating detection"):
        return False
    
    # Step 5: Results Summary
    print_step(5, "RESULTS SUMMARY", "Analyzing pipeline outputs")
    
    try:
        import pandas as pd
        
        # Check training results
        if Path("results").exists():
            model_files = list(Path("results").glob("model_*.joblib"))
            report_files = list(Path("results").glob("report_*.txt"))
            print(f"✅ Trained {len(model_files)} ML models")
            print(f"✅ Generated {len(report_files)} performance reports")
        
        # Check detection results
        if Path("detection_results").exists():
            detection_files = list(Path("detection_results").glob("*.csv"))
            print(f"✅ Generated {len(detection_files)} detection result files")
            
            # Show some detection statistics
            all_detections = next(Path("detection_results").glob("all_detections_*.csv"), None)
            if all_detections:
                df = pd.read_csv(all_detections)
                total = len(df)
                high_conf = df['high_confidence_cheater'].sum()
                print(f"📊 Detection Results: {high_conf}/{total} high-confidence cases ({high_conf/total:.1%})")
        
        # Check generated data
        if Path("data/synthetic_ground_truth.csv").exists():
            gt_df = pd.read_csv("data/synthetic_ground_truth.csv")
            actual_cheaters = gt_df['is_cheater'].sum()
            total_samples = len(gt_df)
            print(f"📊 Generated Data: {actual_cheaters}/{total_samples} cheating cases ({actual_cheaters/total_samples:.1%})")
        
    except Exception as e:
        print(f"⚠️  Could not analyze results: {e}")
    
    print_final_summary(small_dataset, no_plots)
    return True

def print_final_summary(small_dataset, no_plots):
    """Print final summary and next steps"""
    print("\n" + "="*70)
    print("🎉 QUICK START COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📁 Generated Files:")
    print("   📂 data/synthetic_moodle_logs/          - Synthetic Moodle CSV files")
    print("   📂 data/processed_artificial_features_V2.csv - ML-ready features")
    print("   📂 results/                             - Trained models & reports")
    print("   📂 detection_results/                   - Detection outputs")
    print("   📄 cheating_ground_truth.md             - Detailed analysis")
    
    print("\n🔍 What to Check Next:")
    print("   1. Open 'cheating_ground_truth.md' to see synthetic data analysis")
    print("   2. Check 'results/report_*.txt' for model performance details")
    print("   3. Review 'detection_results/*.csv' for detection examples")
    
    if not no_plots:
        print("\n📊 Optional: Generate Visualizations")
        print("   Run: python bab4_viz.py              - Generate analysis plots")
        print("   Run: python analyze_top_offenders.py - Analyze suspicious users")
    
    print("\n📖 For Real Data:")
    print("   1. Place your real Moodle CSV files in 'data/real_moodle_logs/'")
    print("   2. Run preprocessing without --train_mode")
    print("   3. Run cheating_detection.py on real features")
    
    print("\n💡 Performance Summary:")
    if small_dataset:
        print("   - Used small dataset for quick demonstration")
        print("   - Full dataset (800 samples) will give better results")
    else:
        print("   - Used full synthetic dataset (800 samples)")
        print("   - Models should achieve >95% accuracy")
    
    print("\n📚 For More Information:")
    print("   - See README.md for detailed documentation")
    print("   - Check Bab 3 methodology in thesis document")
    print("   - Review individual script help: python script.py --help")

def main():
    parser = argparse.ArgumentParser(description="Quick start for Moodle Cheating Detection")
    parser.add_argument("--small", action="store_true",
                       help="Use smaller dataset for faster execution")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip visualization generation instructions")
    
    args = parser.parse_args()
    
    start_time = time.time()
    success = quick_start_pipeline(small_dataset=args.small, no_plots=args.no_plots)
    total_time = time.time() - start_time
    
    if success:
        print(f"\n⏱️  Total execution time: {total_time:.1f} seconds")
        print("✅ Quick start completed successfully!")
        print("\n🚀 Your Moodle cheating detection system is ready to use!")
    else:
        print(f"\n❌ Quick start failed after {total_time:.1f} seconds")
        print("Check the error messages above and refer to README.md for help")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 