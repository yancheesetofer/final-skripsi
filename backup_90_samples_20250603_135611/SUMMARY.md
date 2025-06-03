# Backup Summary: 90 Samples Training

## Date: 2025-06-03

This backup contains the results from training ML models on a small dataset of 90 samples.

## Dataset Characteristics
- **Total Samples**: 90 (synthetic data)
- **Class Distribution**: 
  - Non-cheaters: 66 (73.3%)
  - Cheaters: 24 (26.7%)
- **Features**: 7 behavioral features
- **Split**: 70% train / 15% validation / 15% test

## Model Performance (Test Set Accuracy)
1. **Random Forest**: 85.71% (Best)
2. **Ensemble (Voting)**: 85.71%
3. **SVM**: 78.57%
4. **Gradient Boosting**: 78.57%
5. **XGBoost**: 78.57%
6. **Neural Network**: 71.43%

## Detection Results on Real Data (446,720 attempts)
Using Ensemble model:
- **High confidence (≥80%)**: 25,309 detections (5.67%)
- **Medium confidence (≥60%)**: 78,409 detections (17.55%)
- **Low confidence (≥40%)**: 99,279 detections (22.22%)

## Key Findings
- Identified 3,049 users with multiple suspicious attempts
- Top offender (User 6023) had 38 suspicious attempts
- Some quizzes showed 100% detection rates

## Limitations
- Very small training set (only 90 samples)
- Model rankings may be unstable
- Need more data for robust conclusions

## Next Steps
- Generate 500-1000 training samples
- Maintain ~27% cheating rate
- Re-train all models
- Compare performance improvements 