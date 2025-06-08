#!/usr/bin/env python3
"""
Comprehensive test script for Moodle Cheating Detection System
Tests the entire pipeline from data generation to model training and detection

Usage: python test_pipeline.py [--quick] [--skip-training]
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
import shutil

class PipelineTest:
    def __init__(self, quick_test=False, skip_training=False):
        self.quick_test = quick_test
        self.skip_training = skip_training
        self.test_results = {}
        self.start_time = time.time()
        
        # Test configuration
        self.test_config = {
            "num_users": 50 if quick_test else 100,
            "num_quizzes": 5 if quick_test else 10,
            "base_date": "2024-01-01",
            "timelimit": 3600
        }
    
    def log(self, message, level="INFO"):
        """Log a message with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def run_command(self, command, description, timeout=300):
        """Run a command and return success status"""
        self.log(f"Running: {description}")
        self.log(f"Command: {command}")
        
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=timeout
            )
            self.log(f"✅ {description} - SUCCESS")
            return True, result.stdout
        except subprocess.TimeoutExpired:
            self.log(f"❌ {description} - TIMEOUT after {timeout}s", "ERROR")
            return False, "Timeout"
        except subprocess.CalledProcessError as e:
            self.log(f"❌ {description} - FAILED", "ERROR")
            self.log(f"Error: {e.stderr}", "ERROR")
            return False, e.stderr
    
    def check_file_exists(self, filepath, description):
        """Check if a file exists and log result"""
        if Path(filepath).exists():
            size = Path(filepath).stat().st_size
            self.log(f"✅ {description} - Found ({size:,} bytes)")
            return True
        else:
            self.log(f"❌ {description} - Not found", "ERROR")
            return False
    
    def test_environment(self):
        """Test 1: Environment and dependencies"""
        self.log("=== TEST 1: Environment Check ===")
        
        # Check Python version
        version = sys.version_info
        if version.major == 3 and 8 <= version.minor <= 10:
            self.log(f"✅ Python version {version.major}.{version.minor}.{version.micro}")
        else:
            self.log(f"❌ Unsupported Python version {version.major}.{version.minor}.{version.micro}", "ERROR")
            return False
        
        # Check critical imports
        critical_packages = [
            'pandas', 'numpy', 'sklearn', 'xgboost', 
            'matplotlib', 'seaborn', 'joblib'
        ]
        
        for package in critical_packages:
            try:
                __import__(package)
                self.log(f"✅ {package} imported successfully")
            except ImportError as e:
                self.log(f"❌ {package} import failed: {e}", "ERROR")
                return False
        
        # Check key files
        key_files = [
            'generate_case.py',
            'enhanced_model.py', 
            'cheating_detection.py',
            'requirements.txt',
            'synthetic_data_config_800.json'
        ]
        
        for file in key_files:
            if not self.check_file_exists(file, f"Key file {file}"):
                return False
        
        self.test_results['environment'] = True
        return True
    
    def test_data_generation(self):
        """Test 2: Synthetic data generation"""
        self.log("=== TEST 2: Data Generation ===")
        
        # Create test config
        config_path = "test_config_pipeline.json"
        with open(config_path, 'w') as f:
            json.dump(self.test_config, f, indent=2)
        
        # Clean up previous test data
        if Path("data/synthetic_moodle_logs").exists():
            shutil.rmtree("data/synthetic_moodle_logs")
        
        # Run data generation
        success, output = self.run_command(
            f"python generate_case.py {config_path}",
            "Generate synthetic data",
            timeout=180
        )
        
        if not success:
            self.test_results['data_generation'] = False
            return False
        
        # Check outputs
        expected_files = [
            "data/synthetic_moodle_logs/mdl_quiz_attempts.csv",
            "data/synthetic_moodle_logs/mdl_question_attempt_steps.csv",
            "data/synthetic_ground_truth.csv",
            "cheating_ground_truth.md"
        ]
        
        for file in expected_files:
            if not self.check_file_exists(file, f"Generated file {file}"):
                self.test_results['data_generation'] = False
                return False
        
        # Clean up test config
        if Path(config_path).exists():
            os.remove(config_path)
        
        self.test_results['data_generation'] = True
        return True
    
    def test_preprocessing(self):
        """Test 3: Data preprocessing and feature extraction"""
        self.log("=== TEST 3: Data Preprocessing ===")
        
        # Clean up previous preprocessing outputs
        if Path("data/processed_artificial_features_V2.csv").exists():
            os.remove("data/processed_artificial_features_V2.csv")
        
        if Path("preprocessing/artifacts").exists():
            shutil.rmtree("preprocessing/artifacts")
            Path("preprocessing/artifacts").mkdir(parents=True, exist_ok=True)
        
        # Run preprocessing
        success, output = self.run_command(
            "python -m preprocessing.main_preprocessor "
            "--data_input_dir data/synthetic_moodle_logs "
            "--output_feature_path data/processed_artificial_features_V2.csv "
            "--train_mode "
            "--ground_truth_path data/synthetic_ground_truth.csv",
            "Preprocess synthetic data",
            timeout=300
        )
        
        if not success:
            self.test_results['preprocessing'] = False
            return False
        
        # Check outputs
        expected_files = [
            "data/processed_artificial_features_V2.csv",
            "preprocessing/artifacts/standard_scaler.joblib",
            "preprocessing/artifacts/feature_imputer.joblib"
        ]
        
        for file in expected_files:
            if not self.check_file_exists(file, f"Preprocessing output {file}"):
                self.test_results['preprocessing'] = False
                return False
        
        # Validate feature file
        try:
            import pandas as pd
            df = pd.read_csv("data/processed_artificial_features_V2.csv")
            
            expected_columns = [
                'max_nav_similarity_zscore', 'mean_nav_similarity_zscore', 
                'median_step_duration', 'nav_revisits_count', 'quick_actions_count',
                'std_nav_similarity_zscore', 'std_step_duration', 'sumgrades', 'is_cheater'
            ]
            
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                self.log(f"❌ Missing columns: {missing_cols}", "ERROR")
                self.test_results['preprocessing'] = False
                return False
            
            self.log(f"✅ Feature file has {len(df)} rows and {len(df.columns)} columns")
            self.log(f"✅ Cheating rate: {df['is_cheater'].mean():.2%}")
            
        except Exception as e:
            self.log(f"❌ Error validating feature file: {e}", "ERROR")
            self.test_results['preprocessing'] = False
            return False
        
        self.test_results['preprocessing'] = True
        return True
    
    def test_model_training(self):
        """Test 4: Model training"""
        if self.skip_training:
            self.log("=== TEST 4: Model Training (SKIPPED) ===")
            self.test_results['model_training'] = 'skipped'
            return True
        
        self.log("=== TEST 4: Model Training ===")
        
        # Clean up previous model outputs
        if Path("results").exists():
            shutil.rmtree("results")
        Path("results").mkdir(exist_ok=True)
        
        # Run model training
        timeout = 300 if self.quick_test else 600
        success, output = self.run_command(
            "python enhanced_model.py",
            "Train ML models",
            timeout=timeout
        )
        
        if not success:
            self.test_results['model_training'] = False
            return False
        
        # Check model outputs
        expected_files = [
            "results/scaler.joblib",
            "results/model_Random Forest.joblib",
            "results/model_Ensemble (Voting).joblib"
        ]
        
        for file in expected_files:
            if not self.check_file_exists(file, f"Model file {file}"):
                self.test_results['model_training'] = False
                return False
        
        # Check performance reports
        report_files = list(Path("results").glob("report_*.txt"))
        if not report_files:
            self.log("❌ No performance reports generated", "ERROR")
            self.test_results['model_training'] = False
            return False
        
        self.log(f"✅ Generated {len(report_files)} performance reports")
        
        self.test_results['model_training'] = True
        return True
    
    def test_detection_simulation(self):
        """Test 5: Detection on synthetic data (simulating real data detection)"""
        self.log("=== TEST 5: Detection Simulation ===")
        
        # Use the same processed features as "real" data for testing
        if not Path("data/processed_artificial_features_V2.csv").exists():
            self.log("❌ No processed features for detection test", "ERROR")
            self.test_results['detection'] = False
            return False
        
        # Copy processed features to real data format for testing
        shutil.copy(
            "data/processed_artificial_features_V2.csv",
            "data/processed_real_features_for_detection_V2.csv"
        )
        
        # Clean up previous detection results
        if Path("detection_results").exists():
            shutil.rmtree("detection_results")
        Path("detection_results").mkdir(exist_ok=True)
        
        # Run detection (skip if no models)
        if not Path("results/scaler.joblib").exists():
            self.log("❌ No trained models for detection test", "ERROR")
            self.test_results['detection'] = False
            return False
        
        success, output = self.run_command(
            "python cheating_detection.py",
            "Run cheating detection",
            timeout=180
        )
        
        if not success:
            self.test_results['detection'] = False
            return False
        
        # Check detection outputs
        detection_files = list(Path("detection_results").glob("*.csv"))
        if not detection_files:
            self.log("❌ No detection results generated", "ERROR")
            self.test_results['detection'] = False
            return False
        
        self.log(f"✅ Generated {len(detection_files)} detection result files")
        
        # Validate detection results
        try:
            import pandas as pd
            results_file = next(Path("detection_results").glob("all_detections_*.csv"))
            df = pd.read_csv(results_file)
            
            required_cols = ['cheating_probability', 'high_confidence_cheater']
            if all(col in df.columns for col in required_cols):
                high_conf_count = df['high_confidence_cheater'].sum()
                self.log(f"✅ Detection results: {high_conf_count} high-confidence detections")
            else:
                self.log("❌ Missing required columns in detection results", "ERROR")
                self.test_results['detection'] = False
                return False
                
        except Exception as e:
            self.log(f"❌ Error validating detection results: {e}", "ERROR")
            self.test_results['detection'] = False
            return False
        
        self.test_results['detection'] = True
        return True
    
    def run_full_test(self):
        """Run the complete test suite"""
        self.log("🚀 Starting Moodle Cheating Detection Pipeline Test")
        self.log(f"Quick test mode: {self.quick_test}")
        self.log(f"Skip training: {self.skip_training}")
        
        tests = [
            ("Environment", self.test_environment),
            ("Data Generation", self.test_data_generation),
            ("Preprocessing", self.test_preprocessing),
            ("Model Training", self.test_model_training),
            ("Detection", self.test_detection_simulation)
        ]
        
        for test_name, test_func in tests:
            try:
                success = test_func()
                if not success and self.test_results.get(test_name.lower().replace(' ', '_')) is not False:
                    self.log(f"❌ {test_name} test failed - stopping pipeline", "ERROR")
                    break
            except Exception as e:
                self.log(f"❌ {test_name} test crashed: {e}", "ERROR")
                self.test_results[test_name.lower().replace(' ', '_')] = False
                break
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        elapsed = time.time() - self.start_time
        
        self.log("=" * 60)
        self.log("🏁 PIPELINE TEST SUMMARY")
        self.log("=" * 60)
        
        passed = sum(1 for v in self.test_results.values() if v is True)
        failed = sum(1 for v in self.test_results.values() if v is False)
        skipped = sum(1 for v in self.test_results.values() if v == 'skipped')
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result is True else "❌ FAIL" if result is False else "⏭️ SKIP"
            self.log(f"{test_name.replace('_', ' ').title()}: {status}")
        
        self.log("-" * 60)
        self.log(f"Total tests: {len(self.test_results)}")
        self.log(f"Passed: {passed}")
        self.log(f"Failed: {failed}")
        self.log(f"Skipped: {skipped}")
        self.log(f"Elapsed time: {elapsed:.1f} seconds")
        
        if failed == 0:
            self.log("🎉 ALL TESTS PASSED! Pipeline is working correctly.")
            return True
        else:
            self.log("💥 SOME TESTS FAILED! Check logs above for details.")
            return False

def main():
    parser = argparse.ArgumentParser(description="Test Moodle Cheating Detection Pipeline")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick test with smaller dataset")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip model training (use existing models)")
    
    args = parser.parse_args()
    
    tester = PipelineTest(quick_test=args.quick, skip_training=args.skip_training)
    success = tester.run_full_test()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 