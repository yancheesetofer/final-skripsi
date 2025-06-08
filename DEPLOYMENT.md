# Deployment Guide: Copy and Run in New Environment

This guide is specifically for **copying the entire system to a new machine/environment** and running it from scratch, simulating what a supervisor or external reviewer would do.

## 🎯 Objective

Successfully copy, install, and run the complete Moodle cheating detection pipeline in a fresh environment without any prior setup.

## 📋 Prerequisites Checklist

Before starting, ensure the target system has:

- [ ] **Python 3.8, 3.9, or 3.10** installed
- [ ] **8GB+ RAM** available
- [ ] **5GB+ free disk space**
- [ ] **Internet connection** for dependency downloads
- [ ] **Command line access** (Terminal/PowerShell)

### Quick Python Version Check
```bash
python --version
# Should show: Python 3.8.x, 3.9.x, or 3.10.x
```

If Python is not installed or wrong version:
- **Windows**: Download from [python.org](https://python.org)
- **macOS**: Use `brew install python@3.9` or download from python.org
- **Linux**: Use `sudo apt install python3.9` or equivalent

## 🚀 Deployment Steps

### Step 1: Copy Project Files

Choose one of these methods:

**Option A: Download/Copy Archive**
```bash
# If you have a ZIP file
unzip moodle-cheating-detection.zip
cd moodle-cheating-detection/

# If you have a tar file
tar -xzf moodle-cheating-detection.tar.gz
cd moodle-cheating-detection/
```

**Option B: Clone Repository**
```bash
git clone <repository-url>
cd moodle-cheating-detection/
```

**Option C: Manual Copy**
```bash
# Copy all files to your target directory
cp -r /source/path/* /target/path/
cd /target/path/
```

### Step 2: Verify File Structure

Ensure you have these critical files:
```bash
ls -la
# Should see:
# README.md
# requirements.txt
# generate_case.py
# enhanced_model.py
# cheating_detection.py
# setup.py
# quick_start.py
# test_pipeline.py
# Makefile
# preprocessing/ directory
# synthetic_data_config_800.json
```

### Step 3: Automated Setup (Recommended)

```bash
# Run the automated setup script
python setup.py

# This will:
# 1. Check Python version
# 2. Create virtual environment
# 3. Guide you through activation
# 4. Install all dependencies
# 5. Create necessary directories
# 6. Validate installation
```

**Follow the on-screen instructions** to activate the virtual environment, then run setup.py again to complete installation.

### Step 4: Manual Setup (Alternative)

If automated setup fails, use manual setup:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create directories
mkdir -p data results detection_results preprocessing/artifacts
```

### Step 5: Quick Validation

Test that everything is working:

```bash
# Quick system test (5-10 minutes)
python test_pipeline.py --quick

# If test passes, try the quick start demo
python quick_start.py --small
```

## 🏃‍♂️ Running the Complete System

### Option 1: One-Command Demo (Easiest)

```bash
# Run complete pipeline with defaults (20-30 minutes)
python quick_start.py

# Or for faster demo with smaller dataset (10-15 minutes)
python quick_start.py --small
```

### Option 2: Step-by-Step Execution

```bash
# 1. Generate synthetic data (2-5 minutes)
python generate_case.py

# 2. Process data and extract features (3-8 minutes)
python -m preprocessing.main_preprocessor \
    --data_input_dir data/synthetic_moodle_logs \
    --output_feature_path data/processed_artificial_features_V2.csv \
    --train_mode \
    --ground_truth_path data/synthetic_ground_truth.csv

# 3. Train machine learning models (10-20 minutes)
python enhanced_model.py

# 4. Run detection on data (2-5 minutes)
python cheating_detection.py
```

### Option 3: Using Makefile (if Make is available)

```bash
# Complete pipeline
make all

# Or step by step
make generate
make preprocess
make train
make detect
```

## 📊 Expected Results

After successful execution, you should see:

### Generated Files
```
data/
├── synthetic_moodle_logs/           # 8 CSV files with synthetic Moodle data
│   ├── mdl_quiz_attempts.csv
│   ├── mdl_question_attempt_steps.csv
│   └── ... (6 more CSV files)
├── processed_artificial_features_V2.csv  # ML-ready features
└── synthetic_ground_truth.csv       # Ground truth labels

results/
├── model_Random Forest.joblib       # Trained Random Forest
├── model_SVM.joblib                # Trained SVM
├── model_Ensemble (Voting).joblib  # Ensemble model
├── scaler.joblib                   # Feature scaler
├── report_*.txt                    # Performance reports
└── confusion_matrix_*.png          # Performance plots

detection_results/
├── all_detections_*.csv            # Complete detection results
├── high_confidence_cheaters_*.csv  # High-confidence cases
└── detection_summary_*.txt         # Summary statistics

Other files:
├── cheating_ground_truth.md        # Detailed data analysis
└── preprocessing/artifacts/        # Saved preprocessing components
```

### Performance Indicators
- **Model Accuracy**: >95% on test set
- **Detection Rate**: 20-30% of samples flagged as suspicious (matches literature)
- **Processing Time**: 30-60 minutes total for 800 samples
- **Memory Usage**: <4GB RAM during training

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue: "Python command not found"
```bash
# Try alternative commands
python3 --version
py --version

# Use the working command throughout
python3 generate_case.py  # instead of python generate_case.py
```

#### Issue: "Permission denied" or "Cannot create directory"
```bash
# Fix permissions (Linux/Mac)
chmod +x *.py
sudo chown -R $USER:$USER .

# Run as administrator (Windows)
# Right-click PowerShell → "Run as Administrator"
```

#### Issue: "Module not found" errors
```bash
# Ensure virtual environment is activated
which python  # Should point to venv/bin/python

# Reinstall requirements
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

#### Issue: Memory errors or slow performance
```bash
# Use smaller dataset for testing
python quick_start.py --small

# Monitor memory usage
top -p $(pgrep -f python)  # Linux/Mac
# Task Manager → Processes → Python  # Windows
```

#### Issue: Models don't train properly
```bash
# Check feature file was created
ls -la data/processed_artificial_features_V2.csv

# Check preprocessing artifacts
ls -la preprocessing/artifacts/

# Try training with verbose output
python enhanced_model.py 2>&1 | tee training.log
```

### Getting Help

1. **Check logs**: Look for error messages in the console output
2. **Validate environment**: Run `python test_pipeline.py --quick`
3. **Check file sizes**: Ensure generated files are not empty
4. **Review README.md**: Contains detailed troubleshooting section

## ✅ Success Validation

Your deployment is successful if:

- [ ] All Python scripts run without errors
- [ ] `results/` directory contains trained models (>1MB each)
- [ ] `detection_results/` contains CSV files with detection outputs
- [ ] Model reports show >90% accuracy
- [ ] `cheating_ground_truth.md` file is generated with analysis

### Final Validation Command
```bash
# Run this to verify everything works
python -c "
import pandas as pd
from pathlib import Path

# Check key files exist
files = [
    'data/processed_artificial_features_V2.csv',
    'results/model_Random Forest.joblib',
    'detection_results'
]

for f in files:
    if Path(f).exists():
        print(f'✅ {f}')
    else:
        print(f'❌ {f}')

# Check data quality
try:
    df = pd.read_csv('data/processed_artificial_features_V2.csv')
    print(f'✅ Features: {len(df)} samples, {len(df.columns)} columns')
    print(f'✅ Cheating rate: {df[\"is_cheater\"].mean():.1%}')
except:
    print('❌ Could not validate feature data')

print('Deployment validation complete!')
"
```

## 🎯 Next Steps After Deployment

1. **Explore Results**: Open `cheating_ground_truth.md` to see data analysis
2. **Check Performance**: Review `results/report_*.txt` files
3. **Understand Detection**: Examine `detection_results/*.csv` files
4. **Try Real Data**: Place real Moodle CSVs in `data/real_moodle_logs/`
5. **Customize**: Modify `synthetic_data_config_800.json` for different scenarios

## 📞 Support

If deployment fails after following this guide:
1. Check that you have the exact Python version requirements (3.8-3.10)
2. Ensure sufficient system resources (8GB RAM, 5GB disk)
3. Verify all files were copied correctly
4. Try the quick validation: `python test_pipeline.py --quick`

The system has been tested on:
- **Windows 10/11** with Python 3.9
- **macOS Big Sur/Monterey** with Python 3.8-3.10  
- **Ubuntu 20.04/22.04** with Python 3.8-3.10

**Expected deployment time**: 45-90 minutes from start to finish, including training.

---

*This deployment guide ensures the system can be copied and run "blindly" in any compatible environment, simulating real-world usage by supervisors or external reviewers.* 