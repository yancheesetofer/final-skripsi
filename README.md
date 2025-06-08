# Moodle Cheating Detection System

**PEMANTAUAN KEPATUHAN SECARA OTOMATIS MELALUI ANALISIS LOG PADA MOODLE BERBASIS KECERDASAN BUATAN**

A comprehensive machine learning pipeline for detecting academic cheating patterns in Moodle log data using artificial intelligence techniques.

## Table of Contents
- [System Overview](#system-overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Directory Structure](#directory-structure)
- [Usage Guide](#usage-guide)
- [Expected Input/Output Formats](#expected-inputoutput-formats)
- [Troubleshooting](#troubleshooting)
- [Technical Documentation](#technical-documentation)

## System Overview

This system implements a machine learning pipeline that:
1. **Generates artificial training data** with controlled cheating patterns
2. **Preprocesses Moodle log data** and extracts behavioral features
3. **Trains ensemble ML models** for cheating detection
4. **Applies trained models** to detect suspicious patterns in real data
5. **Provides comprehensive analysis** and visualization of results

### Key Components
- **Data Generator** (`generate_case.py`): Creates synthetic Moodle logs with ground truth
- **Preprocessing Pipeline** (`preprocessing/`): Transforms raw logs into ML-ready features
- **Model Training** (`enhanced_model.py`): Trains ensemble of ML algorithms
- **Detection System** (`cheating_detection.py`): Applies models to real data
- **Visualization Tools**: Generates comprehensive analysis reports

## System Requirements

### Hardware Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: Minimum 5GB free space
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: Optional (speeds up Neural Network training)

### Software Requirements
- **Python**: 3.8, 3.9, or 3.10 (3.11+ not fully tested)
- **Operating System**: Windows 10+, macOS 10.14+, or Linux Ubuntu 18.04+

### Python Dependencies
See `requirements.txt` for complete list. Key packages:
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

## Installation Guide

### Step 1: Clone/Copy the Project
```bash
# If using git
git clone <repository-url>
cd moodle-cheating-detection

# Or copy all files to your target directory
cp -r /path/to/source/* /your/target/directory/
cd /your/target/directory/
```

### Step 2: Create Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn, xgboost; print('Installation successful!')"
```

### Step 4: Create Required Directories
```bash
# Create necessary directories
mkdir -p data
mkdir -p results
mkdir -p detection_results
mkdir -p preprocessing/artifacts
mkdir -p draft-skripsi-tex/newfigures
```

## Directory Structure

After installation, your directory should look like this:

```
moodle-cheating-detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── generate_case.py                   # Synthetic data generator
├── enhanced_model.py                  # ML model training pipeline
├── cheating_detection.py              # Real data detection system
├── synthetic_data_config_800.json     # Configuration for data generation
├── preprocessing/                     # Data preprocessing modules
│   ├── main_preprocessor.py          # Main preprocessing entry point
│   ├── data_loader.py                # Data loading utilities
│   ├── feature_extractor.py          # Feature extraction logic
│   ├── feature_processor.py          # Feature processing pipeline
│   ├── core_processor.py             # Core data processing functions
│   └── artifacts/                    # Saved preprocessing artifacts
├── data/                             # Data directory (created by you)
│   ├── synthetic_moodle_logs/        # Generated synthetic data
│   ├── real_moodle_logs/            # Real Moodle data (copy from institute_log_legacy/lumbung_sampled/)
│   └── processed_*.csv              # Processed feature files
├── results/                          # Model training outputs
├── detection_results/                # Detection results
└── draft-skripsi-tex/               # LaTeX thesis files
    └── newfigures/                  # Generated figures
```

## Usage Guide

### Phase 1: Generate Synthetic Training Data

**Purpose**: Create artificial Moodle logs with known cheating patterns for model training.

```bash
# Generate 800 synthetic samples with controlled cheating patterns
python generate_case.py

# Output files created:
# - data/synthetic_moodle_logs/*.csv (8 CSV files with synthetic Moodle tables)
# - data/synthetic_ground_truth.csv (labels for each sample)
# - cheating_ground_truth.md (detailed analysis of generated patterns)
```

**Configuration**: Edit `synthetic_data_config_800.json` to adjust:
- Number of users and quizzes
- Cheating group parameters
- Severity levels
- Pattern configurations

### Phase 2: Preprocess Data and Extract Features

**Purpose**: Transform raw log data into machine learning features.

```bash
# For synthetic data (training mode)
python -m preprocessing.main_preprocessor \
    --data_input_dir data/synthetic_moodle_logs \
    --output_feature_path data/processed_artificial_features_V2.csv \
    --train_mode \
    --ground_truth_path data/synthetic_ground_truth.csv

# For real data (detection mode) - if you have real Moodle data
# Note: Copy files from institute_log_legacy/lumbung_sampled/ to data/real_moodle_logs/
python -m preprocessing.main_preprocessor \
    --data_input_dir data/real_moodle_logs \
    --output_feature_path data/processed_real_features_for_detection_V2.csv
```

**Key Processing Steps**:
1. Data loading and validation
2. Timestamp normalization
3. Feature extraction (35 initial features)
4. VIF analysis and feature selection (8 final features)
5. Standardization and artifact saving

### Phase 3: Train Machine Learning Models

**Purpose**: Train ensemble of ML algorithms on synthetic data.

```bash
# Train all models with hyperparameter tuning
python enhanced_model.py

# This will:
# 1. Load processed synthetic features
# 2. Split data (70% train, 15% validation, 15% test)
# 3. Train 6 different models with hyperparameter tuning
# 4. Evaluate performance and save best models
# 5. Generate performance visualizations
```

**Models Trained**:
- Random Forest (with grid search)
- Support Vector Machine (with grid search)
- Gradient Boosting
- XGBoost
- Neural Network (3 layers)
- Ensemble (voting classifier)

**Outputs** (saved in `results/`):
- `model_*.joblib`: Trained models
- `scaler.joblib`: Feature scaler
- `confusion_matrix_*.png`: Performance visualizations
- `curves_*.png`: ROC and PR curves
- `report_*.txt`: Detailed performance reports

### Phase 4: Apply Models to Real Data

**Purpose**: Detect cheating patterns in real Moodle logs.

```bash
# Run cheating detection on real data
python cheating_detection.py

# Requires:
# - data/processed_real_features_for_detection_V2.csv (real data features)
# - results/model_*.joblib (trained models)
# - results/scaler.joblib (feature scaler)
```

**Detection Outputs** (saved in `detection_results/`):
- `all_detections_*.csv`: Complete results with probability scores
- `high_confidence_cheaters_*.csv`: High-confidence detections only
- `detection_summary_*.txt`: Summary statistics

### Phase 5: Analysis and Visualization

**Purpose**: Generate comprehensive analysis of results.

```bash
# Generate analysis visualizations
python bab4_viz.py

# Analyze top offenders
python analyze_top_offenders.py
```

## Expected Input/Output Formats

### Input Data Formats

#### Synthetic Data Generation Input
- `synthetic_data_config_800.json`: Configuration file
```json
{
    "num_users": 800,
    "num_quizzes": 50,
    "base_date": "2024-01-01",
    "cheating_groups": [
        {
            "id": "group_1",
            "severity": "high",
            "size": 10
        }
    ]
}
```

#### Real Moodle Data Input
Required CSV files in `data/real_moodle_logs/`:
*Note: This directory is equivalent to `institute_log_legacy/lumbung_sampled` in the original research data*

- `mdl_quiz_attempts.csv`
- `mdl_question_attempt_steps.csv`
- `mdl_question_attempt_step_data.csv`
- `mdl_quiz.csv`
- `mdl_question_answers.csv`
- `mdl_quiz_grades.csv`
- `mdl_sessions.csv`
- `mdl_question_usages.csv`

### Output Data Formats

#### Processed Features (CSV)
8 columns representing behavioral features:
- `max_nav_similarity_zscore`: Maximum navigation similarity z-score
- `mean_nav_similarity_zscore`: Mean navigation similarity z-score
- `median_step_duration`: Median time per step
- `nav_revisits_count`: Number of question revisits
- `quick_actions_count`: Count of very fast actions
- `std_nav_similarity_zscore`: Standard deviation of navigation similarity
- `std_step_duration`: Standard deviation of step duration
- `sumgrades`: Total grade achieved

#### Detection Results (CSV)
- `attempt_id`, `user_id`, `quiz_id`: Identifiers
- `cheating_probability`: Model confidence (0.0-1.0)
- `high_confidence_cheater`: Binary flag (threshold ≥ 0.8)
- `medium_confidence_cheater`: Binary flag (threshold ≥ 0.6)
- `low_confidence_cheater`: Binary flag (threshold ≥ 0.4)

## Troubleshooting

### Common Installation Issues

#### Issue: `pip install` fails with dependency conflicts
**Solution**:
```bash
# Try installing in fresh environment
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Issue: `ModuleNotFoundError` for installed packages
**Solution**:
```bash
# Verify virtual environment is activated
which python  # Should point to venv/bin/python

# Reinstall specific package
pip uninstall package_name
pip install package_name
```

#### Issue: Memory errors during processing
**Solution**:
```bash
# Process data in smaller chunks
# Edit configuration file to reduce batch size
# Close other applications to free memory
```

### Common Runtime Issues

#### Issue: "File not found" errors
**Solution**:
1. Verify all required directories exist:
```bash
ls -la data/ results/ detection_results/ preprocessing/artifacts/
```
2. Check file paths in error message
3. Ensure previous steps completed successfully

#### Issue: Model training takes too long
**Solution**:
```bash
# Reduce hyperparameter search space in enhanced_model.py
# Use smaller dataset for testing
# Enable parallel processing (set n_jobs=-1)
```

#### Issue: Low detection accuracy
**Solution**:
1. Check feature quality:
```bash
# Verify feature extraction completed without errors
# Check for excessive missing values
# Validate timestamp formats
```
2. Adjust detection thresholds in `cheating_detection.py`
3. Retrain models with different parameters

#### Issue: Inconsistent results across runs
**Solution**:
```bash
# Verify random seeds are set consistently
# Check for data leakage between train/test sets
# Ensure preprocessing artifacts are saved/loaded correctly
```

### Data-Specific Issues

#### Issue: Real Moodle data format mismatch
**Solution**:
1. Compare your CSV headers with expected format
2. Check timestamp formats (should be POSIX timestamps)
3. Verify foreign key relationships between tables
4. Check for required columns in each CSV file

#### Issue: No cheating detected in real data
**Solution**:
1. Check if detection thresholds are too high
2. Verify feature extraction didn't produce all zeros
3. Compare feature distributions between synthetic and real data
4. Consider adjusting model sensitivity

### Performance Optimization

#### For Large Datasets (>100K records):
```bash
# Process in chunks
export CHUNK_SIZE=10000

# Use parallel processing
export N_JOBS=-1

# Monitor memory usage
top -p $(pgrep -f python)
```

#### For Faster Training:
```bash
# Reduce cross-validation folds
# Use smaller hyperparameter grids
# Enable early stopping for Neural Networks
```

## Technical Documentation

### Model Architecture
- **Ensemble Approach**: Combines Random Forest, SVM, Neural Network, and Gradient Boosting
- **Feature Engineering**: 35 initial features reduced to 8 via VIF analysis
- **Evaluation**: Stratified cross-validation with multiple metrics

### Feature Selection Rationale
8 features selected based on:
1. **Low multicollinearity** (VIF < 10)
2. **High predictive importance**
3. **Interpretability** for domain experts
4. **Stability** across different datasets

### Detection Strategy
- **Multi-threshold approach**: High (≥80%), Medium (≥60%), Low (≥40%) confidence
- **Graph analysis**: Network-based group detection
- **Temporal analysis**: Time-based pattern identification

### Validation Approach
- **Synthetic data**: Controlled ground truth for quantitative evaluation
- **Real data**: Large-scale application for qualitative validation
- **Cross-validation**: 5-fold stratified CV for robust performance estimation

---

## Quick Start Checklist

For a rapid deployment, follow this checklist:

- [ ] Python 3.8-3.10 installed
- [ ] Virtual environment created and activated
- [ ] `pip install -r requirements.txt` completed successfully
- [ ] Required directories created (`data/`, `results/`, etc.)
- [ ] Run `python generate_case.py` - should complete without errors
- [ ] Run preprocessing pipeline - creates feature files
- [ ] Run `python enhanced_model.py` - trains models and saves to `results/`
- [ ] If you have real data, run `python cheating_detection.py`
- [ ] Check output files for expected results

**Expected total runtime**: 30-60 minutes for complete pipeline with 800 synthetic samples.

**Success indicators**:
- All Python scripts run without errors
- Model achieves >95% accuracy on test set
- Detection results are generated with reasonable confidence scores
- Visualization files are created in appropriate directories

## Support

If you encounter issues not covered in this troubleshooting guide:
1. Check that all dependencies are correctly installed
2. Verify that all input files are in the expected format
3. Ensure sufficient system resources (RAM, disk space)
4. Review log messages for specific error details

The system has been tested on:
- Windows 10/11 with Python 3.9
- macOS Big Sur/Monterey with Python 3.8-3.10
- Ubuntu 20.04/22.04 with Python 3.8-3.10 