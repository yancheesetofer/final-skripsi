# preprocess_pipeline/feature_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor # Ensure statsmodels is installed
import joblib
import os
import json # For saving list of selected features

ARTIFACTS_DIR = 'artifacts' # Directory to save scalers, imputers, selected features

def handle_missing_values(df, strategy='mean', train_mode=True, imputer_path=None):
    """Handles missing values using SimpleImputer."""
    print(f"Handling missing values with strategy: {strategy}")
    
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)
    
    imputer_path = imputer_path or os.path.join(ARTIFACTS_DIR, 'feature_imputer.joblib')

    # Operate only on numeric columns, ensure original df is not modified unless intended
    numeric_cols = df.select_dtypes(include=np.number).columns
    df_processed = df.copy()

    if not numeric_cols.empty:
        if train_mode:
            imputer = SimpleImputer(strategy=strategy)
            df_processed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            joblib.dump(imputer, imputer_path)
            print(f"Fitted and saved imputer to {imputer_path}")
        else: 
            if os.path.exists(imputer_path):
                imputer = joblib.load(imputer_path)
                df_processed[numeric_cols] = imputer.transform(df[numeric_cols])
                print(f"Loaded and applied imputer from {imputer_path}")
            else:
                print(f"Warning: Imputer not found at {imputer_path}. Imputing with current data's {strategy}.")
                fallback_imputer = SimpleImputer(strategy=strategy)
                df_processed[numeric_cols] = fallback_imputer.fit_transform(df[numeric_cols])
    else:
        print("No numeric columns found to impute.")
            
    return df_processed

def scale_features(df, scaler_type='standard', train_mode=True, scaler_path=None):
    """Scales numerical features."""
    print(f"Scaling features with scaler: {scaler_type}")

    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)

    scaler_path = scaler_path or os.path.join(ARTIFACTS_DIR, f'{scaler_type}_scaler.joblib')
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    df_scaled = df.copy()

    if not numeric_cols.empty:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            print(f"Warning: Unknown scaler type '{scaler_type}'. Not scaling.")
            return df_scaled

        if train_mode:
            df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            joblib.dump(scaler, scaler_path)
            print(f"Fitted and saved {scaler_type} scaler to {scaler_path}")
        else: 
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])
                print(f"Loaded and applied {scaler_type} scaler from {scaler_path}")
            else:
                print(f"Warning: Scaler not found at {scaler_path}. Fitting scaler to current data.")
                df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        print("No numeric columns found to scale.")
            
    return df_scaled

def select_features_by_variance(df, threshold=0.01, train_mode=True, selector_path=None):
    """Removes features with low variance."""
    print(f"Applying variance threshold feature selection (threshold={threshold})...")
    if not os.path.exists(ARTIFACTS_DIR): os.makedirs(ARTIFACTS_DIR)
    selector_path = selector_path or os.path.join(ARTIFACTS_DIR, 'variance_threshold_selector.joblib')
    
    # Operate on numeric columns, excluding IDs if they were accidentally passed
    numeric_df = df.select_dtypes(include=[np.number])
    non_numeric_df = df.select_dtypes(exclude=[np.number])

    if numeric_df.empty:
        print("No numeric features for variance thresholding.")
        return df

    selected_cols = list(numeric_df.columns)

    if train_mode:
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(numeric_df)
        selected_mask = selector.get_support()
        selected_cols = numeric_df.columns[selected_mask].tolist()
        joblib.dump(selector, selector_path)
        # Save the list of selected column names as well
        with open(os.path.join(ARTIFACTS_DIR, 'variance_selected_features.json'), 'w') as f:
            json.dump(selected_cols, f)
        print(f"VarianceThreshold fitted. Selected {len(selected_cols)} features. Selector and feature list saved.")
    else:
        if os.path.exists(selector_path) and os.path.exists(os.path.join(ARTIFACTS_DIR, 'variance_selected_features.json')):
            selector = joblib.load(selector_path)
            with open(os.path.join(ARTIFACTS_DIR, 'variance_selected_features.json'), 'r') as f:
                selected_cols = json.load(f)
            print(f"VarianceThreshold selector and feature list loaded. Applying to select {len(selected_cols)} features.")
        else:
            print("Warning: VarianceThreshold selector or feature list not found. Skipping this selection step in test mode.")
            # In this case, return all numeric columns + non-numeric ones
            return pd.concat([non_numeric_df, numeric_df], axis=1)[df.columns] # Preserve original column order

    return pd.concat([non_numeric_df, numeric_df[selected_cols]], axis=1)[df.columns]


def select_features_by_collinearity(df, vif_threshold=10.0, corr_threshold=0.95, train_mode=True, selected_features_path=None):
    """
    Selects features by addressing multicollinearity.
    In train_mode, it identifies features to remove based on VIF and correlation, then saves the final list.
    In test_mode, it loads the saved list of features and applies it.
    """
    print(f"Applying collinearity feature selection (VIF_threshold={vif_threshold}, Corr_threshold={corr_threshold})...")
    if not os.path.exists(ARTIFACTS_DIR): os.makedirs(ARTIFACTS_DIR)
    selected_features_path = selected_features_path or os.path.join(ARTIFACTS_DIR, 'collinearity_selected_features.json')

    # Separate non-numeric (like IDs) and numeric features
    id_cols_present = [col for col in ['attempt_id', 'user_id', 'quiz_id', 'is_cheater', 'cheating_group_id'] if col in df.columns]
    numeric_df = df.drop(columns=id_cols_present, errors='ignore').select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("No numeric features for collinearity selection.")
        return df

    final_selected_cols = list(numeric_df.columns)

    if train_mode:
        # --- VIF Based Selection (as per assessment recommendations) ---
        # 1. Initial VIF calculation on all numeric features
        # Make sure no NaNs for VIF
        numeric_df_for_vif = numeric_df.dropna()
                if numeric_df_for_vif.shape[0] < 2: # Need at least 2 samples
            print("Warning: Not enough samples after dropna for VIF. Skipping VIF selection.")
            current_selected_cols_after_vif = list(numeric_df.columns)
        else:
            cols_to_check_vif = list(numeric_df_for_vif.columns)
            cols_to_drop_vif = set()

            # Iteratively remove features with high VIF
            # This is a simplified iterative approach; more sophisticated ones exist
            for _ in range(len(cols_to_check_vif) * 2): # Max iterations
                if len(cols_to_check_vif) < 2: break
                
                vif_values = pd.Series(
                    [variance_inflation_factor(numeric_df_for_vif[cols_to_check_vif].values, i) for i in range(len(cols_to_check_vif))],
                    index=cols_to_check_vif
                )
                max_vif = vif_values.max()
                
                # Specific removal based on assessment (VIF = infinity)
                # Example: If 'nav_seq_length' and 'num_actions' are perfectly correlated.
                # The assessment states: "nav_seq_length: ∞ (Perfect correlation), num_actions: ∞ (Perfect correlation)"
                # Let's assume 'num_actions' is generally preferred or 'nav_seq_length' is redundant if 'num_actions' from events exists
                if 'nav_seq_length' in vif_values.index and vif_values['nav_seq_length'] == np.inf:
                    if 'nav_seq_length' in cols_to_check_vif: # Check if not already dropped
                        print(f"Marking 'nav_seq_length' for removal due to VIF=inf.")
                        cols_to_drop_vif.add('nav_seq_length')
                        cols_to_check_vif.remove('nav_seq_length')
                        if len(cols_to_check_vif) < 2: break
                        vif_values = vif_values.drop('nav_seq_length', errors='ignore') # Recalculate next iteration

                # Handle Z-score vs Original based on VIF (Prefer Z-score)
                # If 'X' has high VIF and 'X_zscore' exists, prioritize dropping 'X'
                # This rule is tricky to implement generically without knowing all original/zscore pairs
                # A simpler rule based on assessment: "Keep either original OR z-score features, not both"
                # If a z-score feature exists, and its original also exists AND the original is now among the highest VIFs, drop original.
                if not vif_values.empty: # Check if vif_values is not empty
                    feature_to_drop_this_iteration = vif_values.idxmax()
                    if vif_values[feature_to_drop_this_iteration] > vif_threshold:
                        # Prefer Z-score: if 'X_zscore' is the one with highest VIF, but 'X' also exists, it's complex.
                        # The assessment implies keeping Z-scores for better performance.
                        # If 'feature_X' has high VIF, and 'feature_X_zscore' exists, we should have already chosen to use z-score.
                        # This loop will drop the one with current highest VIF.
                        # If 'total_duration' has VIF 17k and 'total_duration_zscore' exists,
                        # and total_duration is currently the max_vif, it gets dropped.
                        print(f"Marking '{feature_to_drop_this_iteration}' for removal (VIF: {vif_values[feature_to_drop_this_iteration]:.2f})")
                        cols_to_drop_vif.add(feature_to_drop_this_iteration)
                        if feature_to_drop_this_iteration in cols_to_check_vif:
                             cols_to_check_vif.remove(feature_to_drop_this_iteration)
                    else:
                        break # All remaining VIFs are below threshold
                else: break


            print(f"Features dropped based on VIF: {cols_to_drop_vif}")
            current_selected_cols_after_vif = [col for col in numeric_df.columns if col not in cols_to_drop_vif]
            
            # Explicitly handle original vs z-score (if both survived VIF somehow and original is not preferred)
            cols_to_drop_prefer_zscore = set()
            for col in list(current_selected_cols_after_vif): # Iterate over a copy
                if col.endswith("_zscore"):
                    original_col = col.replace("_zscore", "")
                    if original_col in current_selected_cols_after_vif:
                        print(f"Preferring '{col}' over '{original_col}'. Marking '{original_col}' for removal.")
                        cols_to_drop_prefer_zscore.add(original_col)
            
            current_selected_cols_after_vif = [col for col in current_selected_cols_after_vif if col not in cols_to_drop_prefer_zscore]
            print(f"Features dropped due to preferring Z-score versions: {cols_to_drop_prefer_zscore}")

        else: # Not enough samples for VIF
            current_selected_cols_after_vif = list(numeric_df.columns)

        # --- Pairwise Correlation Based Selection ---
        if current_selected_cols_after_vif:
            corr_matrix = numeric_df[current_selected_cols_after_vif].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            cols_to_drop_corr = set()
            for column in upper.columns:
                if column not in cols_to_drop_corr: # Don't check if already marked for drop
                    highly_correlated_with_column = upper[upper[column] > corr_threshold].index
                    for correlated_feature in highly_correlated_with_column:
                        if correlated_feature not in cols_to_drop_corr: # Don't drop if its pair is already dropped
                            # Heuristic: drop the second one in the pair
                            # More advanced: drop based on lower VIF, or less domain importance
                            print(f"Marking '{correlated_feature}' for removal due to high correlation with '{column}' ({upper.loc[correlated_feature, column]:.2f})")
                            cols_to_drop_corr.add(correlated_feature)
            
            print(f"Features dropped based on pairwise correlation (>{corr_threshold}): {cols_to_drop_corr}")
            final_selected_cols = [col for col in current_selected_cols_after_vif if col not in cols_to_drop_corr]
        else:
            final_selected_cols = []

        # Save the list of selected features
        with open(selected_features_path, 'w') as f:
            json.dump(final_selected_cols, f)
        print(f"Selected {len(final_selected_cols)} features after collinearity checks. List saved to {selected_features_path}")

    else: # Test/Detection mode
        if os.path.exists(selected_features_path):
            with open(selected_features_path, 'r') as f:
                final_selected_cols = json.load(f)
            print(f"Loaded list of {len(final_selected_cols)} selected features from {selected_features_path}")
            # Ensure all selected features are present in the current df's numeric part
            missing_in_df = [col for col in final_selected_cols if col not in numeric_df.columns]
            if missing_in_df:
                print(f"Warning: Some selected features not found in current test data: {missing_in_df}. They will be missing.")
                # Add them as NaN or 0 if necessary, or error out depending on strictness
                for col_miss in missing_in_df: numeric_df[col_miss] = 0 # Example: add as 0
            
            # Filter numeric_df to keep only selected_cols and in the correct order
            numeric_df = numeric_df[[col for col in final_selected_cols if col in numeric_df.columns]]

        else:
            print(f"Warning: Selected features list {selected_features_path} not found. Using all available numeric features.")
            # final_selected_cols will be all current numeric_df.columns
            final_selected_cols = list(numeric_df.columns)
            
    # Reconstruct DataFrame with selected numeric features and original non-numeric (ID) columns
    # Ensure original column order as much as possible for the final df for consistency
    retained_cols_ordered = id_cols_present + [col for col in final_selected_cols if col in df.columns] # Keep only valid selected numeric
    
    # Ensure no duplicate columns if id_cols were somehow also in final_selected_cols (unlikely)
    final_df_cols = []
    for col in retained_cols_ordered:
        if col not in final_df_cols:
            final_df_cols.append(col)
            
    return df[final_df_cols].copy() # Return a copy with selected features

if __name__ == '__main__':
    # Assume 'final_features' is the output from feature_extractor.py example
    # Example: Create a dummy DataFrame
    num_samples = 100
    data = {
        'attempt_id': range(num_samples), 'user_id': np.random.randint(1, 10, num_samples), 'quiz_id': np.random.randint(1, 3, num_samples),
        'total_duration': np.random.rand(num_samples) * 1000,
        'total_duration_zscore': np.random.randn(num_samples), # Assume this is the z-score of total_duration
        'num_actions': np.random.rand(num_samples) * 100, # Potentially correlated with total_duration
        'nav_seq_length': (np.random.rand(num_samples) * 100).round(), # Made it distinct from num_actions for testing
        'feature_A': np.random.rand(num_samples),
        'feature_B': np.random.rand(num_samples) * 0.9 + data['feature_A'] * 0.1, # feature_B correlated with feature_A
        'constant_feature': np.ones(num_samples), # Low variance
        'is_cheater': np.random.randint(0,2,num_samples)
    }
    data['num_actions_exact_copy_nav_seq'] = data['nav_seq_length'] # To test VIF infinity

    final_features_dummy = pd.DataFrame(data)
    final_features_dummy.iloc[::10, 3] = np.nan # Introduce some NaNs in total_duration
    
    # --- Training Mode Example ---
    print("--- PROCESSING TRAINING DATA (FEATURE PROCESSOR) ---")
    # Clean artifacts for a fresh run
    if os.path.exists(ARTIFACTS_DIR):
        for f_name in os.listdir(ARTIFACTS_DIR):
            if f_name.endswith((".joblib", ".json")): os.remove(os.path.join(ARTIFACTS_DIR, f_name))

    processed_train_imputed = handle_missing_values(final_features_dummy.copy(), strategy='mean', train_mode=True)
    
    # Collinearity selection before scaling, as scaling doesn't change VIF/correlation ranks
    processed_train_coll = select_features_by_collinearity(processed_train_imputed, vif_threshold=5.0, corr_threshold=0.9, train_mode=True)
    
    # Variance threshold after collinearity
    processed_train_var = select_features_by_variance(processed_train_coll, threshold=0.01, train_mode=True) # 0.01 is common for normalized data
                                                                                                           # If data isn't normalized yet, this might remove too much.
                                                                                                           # Let's apply it after scaling.

    processed_train_scaled = scale_features(processed_train_var, scaler_type='standard', train_mode=True)
    
    # Re-apply variance threshold on scaled data (as threshold=0.01 makes more sense on unit variance data)
    # Or, ensure it's done on data *before* scaling if the original variance matters more.
    # For this example, let's assume it's fine on scaled data for removing near-zero variance post-scaling.
    # However, the assessment implies Variance Threshold is a distinct step. Let's stick to that.
    # The select_features_by_variance needs to be applied carefully. Typically on unscaled data if threshold is absolute,
    # or on scaled if threshold is relative (like 0.01 for features scaled to ~unit variance).
    # For simplicity, we will run variance threshold on the data *before* scaling, then scale the result.
    
    # Corrected order: Impute -> Collinearity -> Variance -> Scale
    processed_train_imputed_again = handle_missing_values(final_features_dummy.copy(), strategy='mean', train_mode=True) # Fresh copy
    processed_train_coll_again = select_features_by_collinearity(processed_train_imputed_again, vif_threshold=5.0, corr_threshold=0.9, train_mode=True)
    
    # Extract numeric cols for variance threshold from the result of collinearity selection
    numeric_cols_for_var = [col for col in processed_train_coll_again.columns if col not in ['attempt_id', 'user_id', 'quiz_id', 'is_cheater', 'cheating_group_id']]
    numeric_part_for_var = processed_train_coll_again[numeric_cols_for_var]
    ids_part_for_var = processed_train_coll_again[['attempt_id', 'user_id', 'quiz_id', 'is_cheater', 'cheating_group_id'] if 'is_cheater' in processed_train_coll_again else ['attempt_id', 'user_id', 'quiz_id']]


    processed_train_var_applied = select_features_by_variance(numeric_part_for_var, threshold=0.01, train_mode=True) # This now returns only selected numeric
    processed_train_var_recombined = pd.concat([ids_part_for_var.reset_index(drop=True), processed_train_var_applied.reset_index(drop=True)], axis=1)

    scaled_features_train_final = scale_features(processed_train_var_recombined, scaler_type='standard', train_mode=True)
    
    print("\nFinal Scaled Training Features Sample (after all selections):")
    print(scaled_features_train_final.head())
    scaled_features_train_final.info()


    # --- Test/Detection Mode Example (using saved artifacts) ---
    print("\n--- PROCESSING TEST DATA (FEATURE PROCESSOR - SIMULATED) ---")
    test_data_dummy = final_features_dummy.sample(n=20, random_state=42).copy() # Sample to simulate test data
    test_data_dummy.iloc[::5, 3] = np.nan # Introduce some NaNs in total_duration

    processed_test_imputed = handle_missing_values(test_data_dummy.copy(), strategy='mean', train_mode=False)
    processed_test_coll = select_features_by_collinearity(processed_test_imputed, train_mode=False) # Thresholds not needed, uses saved list
    
    # Apply variance selection using saved selector (or rather, saved list of columns)
    # The select_features_by_variance in test mode now uses the saved list of features.
    numeric_cols_for_var_test = [col for col in processed_test_coll.columns if col not in ['attempt_id', 'user_id', 'quiz_id', 'is_cheater', 'cheating_group_id']]
    numeric_part_for_var_test = processed_test_coll[numeric_cols_for_var_test]
    ids_part_for_var_test = processed_test_coll[['attempt_id', 'user_id', 'quiz_id', 'is_cheater', 'cheating_group_id'] if 'is_cheater' in processed_test_coll else ['attempt_id', 'user_id', 'quiz_id']]

    processed_test_var_applied = select_features_by_variance(numeric_part_for_var_test, train_mode=False) # Uses saved list
    processed_test_var_recombined = pd.concat([ids_part_for_var_test.reset_index(drop=True), processed_test_var_applied.reset_index(drop=True)], axis=1)
    
    scaled_features_test_final = scale_features(processed_test_var_recombined, scaler_type='standard', train_mode=False)
    
    print("\nFinal Scaled Test Features Sample (after all selections):")
    print(scaled_features_test_final.head())
    scaled_features_test_final.info()