# Machine Learning Pipeline for Cheating Detection

## Overview

This pipeline consists of two main scripts that implement a scientifically rigorous approach to detect cheating in online assessments:

1. **`enhanced_model.py`** - Trains and evaluates machine learning models
2. **`cheating_detection.py`** - Applies trained models to detect cheating in real data

## Prerequisites

```bash
# Required packages
pip install pandas numpy scikit-learn xgboost tensorflow joblib matplotlib seaborn
```

## Directory Structure

```
project/
├── data/
│   ├── processed_artificial_features_V2.csv      # Training data (with labels)
│   └── processed_real_features_for_detection_V2.csv  # Real data for detection
├── results/                                      # Created by enhanced_model.py
│   ├── model_*.joblib                           # Saved models
│   ├── model_Neural Network.h5                  # Saved neural network
│   ├── scaler.joblib                           # Feature scaler
│   ├── confusion_matrix_*.png                  # Evaluation plots
│   ├── curves_*.png                            # ROC and PR curves
│   ├── report_*.txt                            # Classification reports
│   └── feature_importance_*.png                # Feature importance plot
├── detection_results/                           # Created by cheating_detection.py
│   ├── all_detections_*.csv                    # Full detection results
│   ├── high_confidence_cheaters_*.csv          # High-confidence detections only
│   └── detection_summary_*.txt                 # Summary statistics
├── enhanced_model.py
├── cheating_detection.py
└── README_ML_Pipeline.md
```

## Step 1: Train Models

Run the training script:

```bash
python enhanced_model.py
```

This will:
- Split data into train (70%), validation (15%), and test (15%) sets
- Train 5 models: Random Forest, SVM, Gradient Boosting, XGBoost, and Neural Network
- Create an ensemble model using soft voting
- Perform hyperparameter tuning for RF and SVM
- Evaluate all models and save results to the `results/` folder
- Generate feature importance analysis

### Expected Outputs

1. **Classification Reports** (`report_*.txt`): Performance metrics for each model
2. **Confusion Matrices** (`confusion_matrix_*.png`): Visual error analysis
3. **ROC & PR Curves** (`curves_*.png`): Threshold analysis plots
4. **Feature Importance** (`feature_importance_Random Forest.png`): Which features matter most
5. **Saved Models** (`model_*.joblib/.h5`): Trained models for detection

## Step 2: Detect Cheating in Real Data

After training, run the detection script:

```bash
python cheating_detection.py
```

This will:
- Load the 446,720 real exam attempts
- Apply the ensemble model (or best available model)
- Classify attempts into confidence levels:
  - High confidence (≥80% probability)
  - Medium confidence (≥60% probability)
  - Low confidence (≥40% probability)
- Generate comprehensive detection reports

### Expected Outputs

1. **Full Results** (`all_detections_*.csv`): All attempts with probability scores
2. **High-Confidence List** (`high_confidence_cheaters_*.csv`): Likely cheaters
3. **Summary Report** (`detection_summary_*.txt`): Statistical overview

## Interpreting Results

### Model Performance (from enhanced_model.py)

Look for in console output:
```
🏆 Best Model (by accuracy): [Model Name] with Accuracy: X.XXXX
```

Check `report_Ensemble (Voting).txt` vs individual model reports to quantify ensemble improvement.

### Detection Results (from cheating_detection.py)

The summary will show:
```
=== Detection Results Summary ===
Total attempts analyzed: 446,720

Cheating detections by confidence level:
- High confidence (≥80%): X,XXX (X.XX%)
- Medium confidence (≥60%): X,XXX (X.XX%)
- Low confidence (≥40%): X,XXX (X.XX%)
```

### Key Metrics to Report

1. **Model Accuracy**: From test set evaluation
2. **Ensemble Improvement**: Compare ensemble vs individual models
3. **Detection Rate**: Percentage of attempts flagged as suspicious
4. **Feature Importance**: Which behavioral patterns are most indicative

## Customization

### Adjusting Detection Thresholds

In `cheating_detection.py`, modify:
```python
DETECTION_THRESHOLDS = {
    'high_confidence': 0.8,    # Adjust based on desired strictness
    'medium_confidence': 0.6,
    'low_confidence': 0.4
}
```

### Using a Different Model for Detection

Change the model choice in the main() function:
```python
model_choice = 'Random Forest'  # or any other trained model
```

## Troubleshooting

1. **"Scaler not found"**: Run `enhanced_model.py` first
2. **"Model not found"**: Ensure `enhanced_model.py` completed successfully
3. **Memory issues**: Process data in batches for very large datasets

## Scientific Rigor

This pipeline implements:
- Proper train/validation/test splits with stratification
- Cross-validation for hyperparameter tuning
- Multiple evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Feature scaling to ensure fair model comparison
- Ensemble methods to improve robustness
- Comprehensive error analysis through confusion matrices

## For Your Thesis

Use these outputs for Bab 4.2:
- Feature importance plot → "What patterns indicate cheating?"
- Model comparison table → "How much does ensemble improve accuracy?"
- Detection statistics → "Real-world application results" 