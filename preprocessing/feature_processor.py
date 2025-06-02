# preprocess_pipeline/feature_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib # For saving/loading scalers and imputers
import os

# --- Thesis Section 3.6.5: Pra-pemrosesan Fitur untuk Kompatibilitas Model ---
# --- Thesis Section 3.6.4: Pemeriksaan Multikolinearitas ---

ARTIFACTS_DIR = 'artifacts' # Directory to save scalers, imputers

def handle_missing_values(df, strategy='mean', train_mode=True, imputer_path=None):
    """Handles missing values using SimpleImputer."""
    # PROFESSOR_REVIEW: Justification for imputation strategy (mean, median, constant).
    # Mean is common, but median is more robust to outliers. Constant can be used if 0 has specific meaning.
    print(f"Handling missing values with strategy: {strategy}")
    
    # Ensure artifacts directory exists
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)
    
    if imputer_path is None: # Default path if not provided
        imputer_path = os.path.join(ARTIFACTS_DIR, 'feature_imputer.joblib')

    numeric_cols = df.select_dtypes(include=np.number).columns
    
    if train_mode:
        imputer = SimpleImputer(strategy=strategy)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        joblib.dump(imputer, imputer_path)
        print(f"Fitted and saved imputer to {imputer_path}")
    else: # Test/Detection mode
        if os.path.exists(imputer_path):
            imputer = joblib.load(imputer_path)
            df[numeric_cols] = imputer.transform(df[numeric_cols])
            print(f"Loaded and applied imputer from {imputer_path}")
        else:
            # Fallback if imputer not found: impute with current data's mean (less ideal for consistency)
            print(f"Warning: Imputer not found at {imputer_path}. Imputing with current data's {strategy}.")
            # This path should ideally not be taken in production to ensure perfect consistency.
            fallback_imputer = SimpleImputer(strategy=strategy)
            df[numeric_cols] = fallback_imputer.fit_transform(df[numeric_cols])
            
    return df

def scale_features(df, scaler_type='standard', train_mode=True, scaler_path=None):
    """Scales numerical features."""
    # PROFESSOR_REVIEW: Justification for scaler choice (StandardScaler vs MinMaxScaler).
    # StandardScaler for features assumed to be normally distributed or for models sensitive to feature variance.
    # MinMaxScaler for neural networks or when features need to be in a specific range [0,1].
    print(f"Scaling features with scaler: {scaler_type}")

    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)

    if scaler_path is None: # Default path
         scaler_path = os.path.join(ARTIFACTS_DIR, f'{scaler_type}_scaler.joblib')
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        print(f"Warning: Unknown scaler type '{scaler_type}'. Not scaling.")
        return df

    if train_mode:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        joblib.dump(scaler, scaler_path)
        print(f"Fitted and saved {scaler_type} scaler to {scaler_path}")
    else: # Test/Detection mode
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            df[numeric_cols] = scaler.transform(df[numeric_cols])
            print(f"Loaded and applied {scaler_type} scaler from {scaler_path}")
        else:
            # Fallback: fit_transform on current data (less ideal for consistency)
            print(f"Warning: Scaler not found at {scaler_path}. Fitting scaler to current data.")
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            
    return df

def calculate_vif(df_features):
    """Calculates Variance Inflation Factor for features."""
    # This is more for analysis than a transformation step.
    # Output can be logged. Feature removal based on VIF is a modeling choice.
    # --- Thesis Section 3.6.4: Pemeriksaan Multikolinearitas ---
    from statsmodels.stats.outliers_influence import variance_inflation_factor # statsmodels package
    
    numeric_df = df_features.select_dtypes(include=[np.number])
    # VIF requires a constant term for intercept if not already centered.
    # For simplicity, we assume features are ready or that VIF is calculated on scaled data.
    # Dropping rows with NaNs for VIF calculation as it can't handle them.
    numeric_df_no_na = numeric_df.dropna()
    if numeric_df_no_na.empty:
        print("Warning: No data left after dropping NaNs for VIF calculation.")
        return pd.DataFrame(columns=['feature', 'VIF'])

    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_df_no_na.columns
    try:
        vif_data["VIF"] = [variance_inflation_factor(numeric_df_no_na.values, i) for i in range(len(numeric_df_no_na.columns))]
        print("\nVIF Scores (Top 10):")
        print(vif_data.sort_values('VIF', ascending=False).head(10))
        # PROFESSOR_REVIEW: High VIF (e.g., >5 or >10) suggests multicollinearity.
        # Decision to remove features based on VIF should be justified.
        # Some models (like Random Forest) are less sensitive to multicollinearity than regression models.
        if not os.path.exists(ARTIFACTS_DIR): os.makedirs(ARTIFACTS_DIR)
        vif_data.to_csv(os.path.join(ARTIFACTS_DIR, 'vif_scores.csv'), index=False)
    except Exception as e:
        print(f"Error calculating VIF: {e}. Potentially due to perfect multicollinearity or all-NaN columns after dropna.")
        return pd.DataFrame(columns=['feature', 'VIF'])
        
    return vif_data


if __name__ == '__main__':
    # Assume 'final_features' is the output from feature_extractor.py example
    # This is a placeholder, you'd load the actual DataFrame
    # final_features = pd.read_csv('path_to_output_of_feature_extractor.csv') 

    # Example: Create a dummy DataFrame similar to what feature_extractor might output
    num_samples = 100
    data = {
        'attempt_id': range(num_samples),
        'user_id': np.random.randint(1, 10, num_samples),
        'quiz_id': np.random.randint(1, 3, num_samples),
        'total_duration': np.random.rand(num_samples) * 1000 + np.nan, # Introduce some NaNs
        'num_actions': np.random.randint(10, 50, num_samples),
        'mean_nav_similarity': np.random.rand(num_samples),
        'total_duration_zscore': np.random.randn(num_samples)
    }
    data['total_duration'][::10] = np.nan # Make 10% NaN
    final_features_dummy = pd.DataFrame(data)
    
    # --- Training Mode Example ---
    print("--- PROCESSING TRAINING DATA ---")
    processed_features_train = handle_missing_values(final_features_dummy.copy(), strategy='mean', train_mode=True)
    scaled_features_train = scale_features(processed_features_train, scaler_type='standard', train_mode=True)
    vif_scores_train = calculate_vif(scaled_features_train.drop(columns=['attempt_id', 'user_id', 'quiz_id'], errors='ignore'))
    # print("\nScaled Training Features Sample:")
    # print(scaled_features_train.head())

    # --- Test/Detection Mode Example (using saved artifacts) ---
    print("\n--- PROCESSING TEST DATA (SIMULATED) ---")
    # Simulate new data that needs processing
    test_data_dummy = pd.DataFrame({
        'attempt_id': range(num_samples, num_samples + 50),
        'user_id': np.random.randint(1, 10, 50),
        'quiz_id': np.random.randint(1, 3, 50),
        'total_duration': np.random.rand(50) * 1100, # Slightly different distribution
        'num_actions': np.random.randint(15, 55, 50),
        'mean_nav_similarity': np.random.rand(50),
        'total_duration_zscore': np.random.randn(50)
    })
    test_data_dummy['total_duration'][::5] = np.nan

    processed_features_test = handle_missing_values(test_data_dummy.copy(), strategy='mean', train_mode=False)
    scaled_features_test = scale_features(processed_features_test, scaler_type='standard', train_mode=False)
    # print("\nScaled Test Features Sample:")
    # print(scaled_features_test.head())