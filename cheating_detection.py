# cheating_detection.py

import pandas as pd
import numpy as np
from joblib import load
# No need for TensorFlow anymore
import os
from datetime import datetime

# --- Configuration ---

# File Paths
REAL_DATA_PATH = 'data/processed_real_features_for_detection_V2.csv'
MODELS_DIR = 'results'
OUTPUT_DIR = 'detection_results'

# Feature columns (must match training)
FEATURE_COLUMNS = [
    'max_nav_similarity_zscore', 'mean_nav_similarity_zscore', 'median_step_duration', 
    'nav_revisits_count_zscore', 'quick_actions_count', 'std_step_duration', 'sumgrades'
]

# ID columns for tracking
ID_COLUMNS = ['attempt_id', 'user_id', 'quiz_id']

# Model files (all are now .joblib)
MODEL_FILES = {
    'Random Forest': 'model_Random Forest.joblib',
    'SVM': 'model_SVM.joblib',
    'Gradient Boosting': 'model_Gradient Boosting.joblib',
    'XGBoost': 'model_XGBoost.joblib',
    'Neural Network': 'model_Neural Network.joblib',
    'Ensemble (Voting)': 'model_Ensemble (Voting).joblib'
}

# Detection thresholds (can be adjusted based on requirements)
DETECTION_THRESHOLDS = {
    'high_confidence': 0.8,    # 80% probability threshold
    'medium_confidence': 0.6,  # 60% probability threshold
    'low_confidence': 0.4      # 40% probability threshold
}


def load_real_data():
    """Load and prepare real data for detection."""
    print("Loading real data for detection...")
    
    if not os.path.exists(REAL_DATA_PATH):
        raise FileNotFoundError(f"Real data file not found at {REAL_DATA_PATH}")
    
    df = pd.read_csv(REAL_DATA_PATH)
    print(f"Loaded {len(df):,} attempts from real data")
    
    # Separate features and IDs
    X = df[FEATURE_COLUMNS]
    ids = df[ID_COLUMNS]
    
    # Load and apply the same scaler used in training
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run enhanced_model.py first.")
    
    scaler = load(scaler_path)
    X_scaled = scaler.transform(X)
    
    return X_scaled, ids, df


def load_trained_models():
    """Load all trained models."""
    print("\nLoading trained models...")
    models = {}
    
    for model_name, filename in MODEL_FILES.items():
        model_path = os.path.join(MODELS_DIR, filename)
        
        if not os.path.exists(model_path):
            print(f"Warning: {model_name} not found at {model_path}. Skipping...")
            continue
        
        try:
            models[model_name] = load(model_path)
            print(f"✓ Loaded {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    return models


def detect_cheating(X, models, model_choice='Ensemble (Voting)'):
    """
    Detect cheating using specified model.
    Returns probability scores and binary predictions.
    """
    if model_choice not in models:
        raise ValueError(f"Model '{model_choice}' not available. Available models: {list(models.keys())}")
    
    print(f"\nRunning detection using {model_choice}...")
    model = models[model_choice]
    
    # Get probability predictions (all models now use the same interface)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Create multi-level predictions based on thresholds
    predictions = pd.DataFrame({
        'cheating_probability': probabilities,
        'high_confidence_cheater': (probabilities >= DETECTION_THRESHOLDS['high_confidence']).astype(int),
        'medium_confidence_cheater': (probabilities >= DETECTION_THRESHOLDS['medium_confidence']).astype(int),
        'low_confidence_cheater': (probabilities >= DETECTION_THRESHOLDS['low_confidence']).astype(int)
    })
    
    return predictions


def analyze_detection_results(predictions, ids):
    """Analyze and summarize detection results."""
    print("\n=== Detection Results Summary ===")
    
    total_attempts = len(predictions)
    
    # Overall statistics
    high_conf = predictions['high_confidence_cheater'].sum()
    medium_conf = predictions['medium_confidence_cheater'].sum()
    low_conf = predictions['low_confidence_cheater'].sum()
    
    print(f"\nTotal attempts analyzed: {total_attempts:,}")
    print(f"\nCheating detections by confidence level:")
    print(f"- High confidence (≥{DETECTION_THRESHOLDS['high_confidence']:.0%}): {high_conf:,} ({high_conf/total_attempts:.2%})")
    print(f"- Medium confidence (≥{DETECTION_THRESHOLDS['medium_confidence']:.0%}): {medium_conf:,} ({medium_conf/total_attempts:.2%})")
    print(f"- Low confidence (≥{DETECTION_THRESHOLDS['low_confidence']:.0%}): {low_conf:,} ({low_conf/total_attempts:.2%})")
    
    # Combine with IDs for detailed analysis
    results_df = pd.concat([ids, predictions], axis=1)
    
    # Analysis by exam
    print("\n--- Detections by Exam (High Confidence) ---")
    exam_summary = results_df.groupby('quiz_id').agg({
        'high_confidence_cheater': ['sum', 'count', 'mean']
    }).round(4)
    exam_summary.columns = ['Cheaters_Detected', 'Total_Attempts', 'Cheating_Rate']
    print(exam_summary.sort_values('Cheating_Rate', ascending=False).head(10))
    
    # Find users with multiple suspicious attempts
    print("\n--- Users with Multiple High-Confidence Detections ---")
    user_summary = results_df[results_df['high_confidence_cheater'] == 1].groupby('user_id').size()
    repeat_cheaters = user_summary[user_summary > 1].sort_values(ascending=False)
    if len(repeat_cheaters) > 0:
        print(f"Found {len(repeat_cheaters)} users with multiple suspicious attempts")
        print(repeat_cheaters.head(10))
    else:
        print("No users found with multiple high-confidence detections")
    
    return results_df


def save_detection_results(results_df, model_name):
    """Save detection results to CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all results
    all_results_path = os.path.join(OUTPUT_DIR, f'all_detections_{model_name}_{timestamp}.csv')
    results_df.to_csv(all_results_path, index=False)
    print(f"\nAll results saved to: {all_results_path}")
    
    # Save high-confidence detections separately
    high_conf_df = results_df[results_df['high_confidence_cheater'] == 1]
    if len(high_conf_df) > 0:
        high_conf_path = os.path.join(OUTPUT_DIR, f'high_confidence_cheaters_{model_name}_{timestamp}.csv')
        high_conf_df.to_csv(high_conf_path, index=False)
        print(f"High-confidence detections saved to: {high_conf_path}")
    
    # Create summary report
    summary_path = os.path.join(OUTPUT_DIR, f'detection_summary_{model_name}_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Cheating Detection Summary\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Attempts: {len(results_df):,}\n\n")
        
        f.write("Detection Statistics:\n")
        for conf_level, threshold in DETECTION_THRESHOLDS.items():
            count = results_df[f'{conf_level}_cheater'].sum()
            f.write(f"- {conf_level.replace('_', ' ').title()} (≥{threshold:.0%}): {count:,} ({count/len(results_df):.2%})\n")
        
        f.write(f"\nProbability Statistics:\n")
        f.write(f"- Mean: {results_df['cheating_probability'].mean():.4f}\n")
        f.write(f"- Std: {results_df['cheating_probability'].std():.4f}\n")
        f.write(f"- Min: {results_df['cheating_probability'].min():.4f}\n")
        f.write(f"- Max: {results_df['cheating_probability'].max():.4f}\n")
        f.write(f"- Median: {results_df['cheating_probability'].median():.4f}\n")
    
    print(f"Summary report saved to: {summary_path}")


def main():
    """Main function to run cheating detection on real data."""
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Load data and models
    X_scaled, ids, original_df = load_real_data()
    models = load_trained_models()
    
    # Check if ensemble model is available, otherwise use best individual model
    if 'Ensemble (Voting)' in models:
        model_choice = 'Ensemble (Voting)'
    elif 'Random Forest' in models:
        model_choice = 'Random Forest'
        print("\nNote: Ensemble model not found. Using Random Forest instead.")
    else:
        model_choice = list(models.keys())[0]
        print(f"\nNote: Using {model_choice} for detection.")
    
    # Run detection
    predictions = detect_cheating(X_scaled, models, model_choice)
    
    # Analyze results
    results_df = analyze_detection_results(predictions, ids)
    
    # Save results
    save_detection_results(results_df, model_choice)
    
    print("\n✅ Detection complete! Check the 'detection_results' folder for outputs.")
    
    # Optional: Run detection with other models for comparison
    print("\n--- Running detection with all available models for comparison ---")
    comparison_results = {}
    
    for model_name in models.keys():
        if model_name != model_choice:  # Skip the one we already ran
            try:
                pred = detect_cheating(X_scaled, models, model_name)
                high_conf_count = pred['high_confidence_cheater'].sum()
                comparison_results[model_name] = high_conf_count
                print(f"{model_name}: {high_conf_count:,} high-confidence detections")
            except Exception as e:
                print(f"Error with {model_name}: {e}")
    
    return results_df


if __name__ == '__main__':
    main() 