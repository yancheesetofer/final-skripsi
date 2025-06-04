# preprocess_pipeline/feature_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
import os
import json

ARTIFACTS_DIR = 'artifacts'

def handle_missing_values(df, strategy='mean', train_mode=True, imputer_path=None):
    """Handles missing values using SimpleImputer."""
    print(f"Handling missing values with strategy: {strategy}")
    
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)
    
    imputer_path = imputer_path or os.path.join(ARTIFACTS_DIR, 'feature_imputer.joblib')

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
    if not os.path.exists(ARTIFACTS_DIR): 
        os.makedirs(ARTIFACTS_DIR)
    selector_path = selector_path or os.path.join(ARTIFACTS_DIR, 'variance_threshold_selector.joblib')
    
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
            return pd.concat([non_numeric_df, numeric_df], axis=1)[df.columns]

    return pd.concat([non_numeric_df, numeric_df[selected_cols]], axis=1)[df.columns]

def select_features_by_collinearity(df, vif_threshold=10.0, corr_threshold=0.95, train_mode=True, selected_features_path=None):
    """
    Selects features by addressing multicollinearity.
    In train_mode, it identifies features to remove based on VIF and correlation, then saves the final list.
    In test_mode, it loads the saved list of features and applies it.
    """
    print(f"Applying collinearity feature selection (VIF_threshold={vif_threshold}, Corr_threshold={corr_threshold})...")
    if not os.path.exists(ARTIFACTS_DIR): 
        os.makedirs(ARTIFACTS_DIR)
    selected_features_path = selected_features_path or os.path.join(ARTIFACTS_DIR, 'collinearity_selected_features.json')

    # Separate non-numeric (like IDs) and numeric features
    id_cols_present = [col for col in ['attempt_id', 'user_id', 'quiz_id', 'is_cheater', 'cheating_group_id'] if col in df.columns]
    numeric_df = df.drop(columns=id_cols_present, errors='ignore').select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("No numeric features for collinearity selection.")
        return df

    final_selected_cols = list(numeric_df.columns)

    if train_mode:
        # VIF Based Selection
        numeric_df_for_vif = numeric_df.dropna()
        if numeric_df_for_vif.shape[0] < 2:
            print("Warning: Not enough samples after dropna for VIF. Skipping VIF selection.")
            current_selected_cols_after_vif = list(numeric_df.columns)
        else:
            cols_to_check_vif = list(numeric_df_for_vif.columns)
            cols_to_drop_vif = set()

            # Iteratively remove features with high VIF
            for _ in range(len(cols_to_check_vif) * 2):
                if len(cols_to_check_vif) < 2: 
                    break
                
                try:
                    vif_values = pd.Series(
                        [variance_inflation_factor(numeric_df_for_vif[cols_to_check_vif].values, i) for i in range(len(cols_to_check_vif))],
                        index=cols_to_check_vif
                    )
                except Exception as e:
                    print(f"VIF calculation failed: {e}. Stopping VIF selection.")
                    break
                
                # Handle infinity VIF values (perfect correlation)
                inf_features = vif_values[vif_values == np.inf].index.tolist()
                for inf_feature in inf_features:
                    print(f"Marking '{inf_feature}' for removal due to VIF=inf.")
                    cols_to_drop_vif.add(inf_feature)
                    if inf_feature in cols_to_check_vif:
                        cols_to_check_vif.remove(inf_feature)

                if len(cols_to_check_vif) < 2:
                    break

                # Recalculate VIF after removing infinity features
                if cols_to_check_vif:
                    try:
                        vif_values = pd.Series(
                            [variance_inflation_factor(numeric_df_for_vif[cols_to_check_vif].values, i) for i in range(len(cols_to_check_vif))],
                            index=cols_to_check_vif
                        )
                    except Exception as e:
                        print(f"VIF recalculation failed: {e}. Stopping VIF selection.")
                        break

                    if not vif_values.empty:
                        max_vif_feature = vif_values.idxmax()
                        max_vif_value = vif_values[max_vif_feature]
                        
                        if max_vif_value > vif_threshold:
                            print(f"Marking '{max_vif_feature}' for removal (VIF: {max_vif_value:.2f})")
                            cols_to_drop_vif.add(max_vif_feature)
                            if max_vif_feature in cols_to_check_vif:
                                cols_to_check_vif.remove(max_vif_feature)
                        else:
                            break
                    else:
                        break
                else:
                    break

            print(f"Features dropped based on VIF: {cols_to_drop_vif}")
            current_selected_cols_after_vif = [col for col in numeric_df.columns if col not in cols_to_drop_vif]
            
            # Handle original vs z-score preference
            cols_to_drop_prefer_zscore = set()
            for col in list(current_selected_cols_after_vif):
                if col.endswith("_zscore"):
                    original_col = col.replace("_zscore", "")
                    if original_col in current_selected_cols_after_vif:
                        print(f"Preferring '{col}' over '{original_col}'. Marking '{original_col}' for removal.")
                        cols_to_drop_prefer_zscore.add(original_col)
            
            current_selected_cols_after_vif = [col for col in current_selected_cols_after_vif if col not in cols_to_drop_prefer_zscore]
            print(f"Features dropped due to preferring Z-score versions: {cols_to_drop_prefer_zscore}")

        # Pairwise Correlation Based Selection
        if current_selected_cols_after_vif:
            corr_matrix = numeric_df[current_selected_cols_after_vif].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            cols_to_drop_corr = set()
            
            for column in upper.columns:
                if column not in cols_to_drop_corr:
                    highly_correlated = upper[upper[column] > corr_threshold].index
                    for correlated_feature in highly_correlated:
                        if correlated_feature not in cols_to_drop_corr:
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

    else:
        # Test/Detection mode
        if os.path.exists(selected_features_path):
            with open(selected_features_path, 'r') as f:
                final_selected_cols = json.load(f)
            print(f"Loaded list of {len(final_selected_cols)} selected features from {selected_features_path}")
            
            missing_in_df = [col for col in final_selected_cols if col not in numeric_df.columns]
            if missing_in_df:
                print(f"Warning: Some selected features not found in current test data: {missing_in_df}")
                for col_miss in missing_in_df: 
                    numeric_df[col_miss] = 0
            
            numeric_df = numeric_df[[col for col in final_selected_cols if col in numeric_df.columns]]
        else:
            print(f"Warning: Selected features list {selected_features_path} not found. Using all available numeric features.")
            final_selected_cols = list(numeric_df.columns)
            
    # Reconstruct DataFrame
    retained_cols_ordered = id_cols_present + [col for col in final_selected_cols if col in df.columns]
    
    final_df_cols = []
    for col in retained_cols_ordered:
        if col not in final_df_cols:
            final_df_cols.append(col)
            
    return df[final_df_cols].copy()

def calculate_vif(df):
    """Calculate VIF for features (legacy function for compatibility)."""
    print("Calculating VIF scores for analysis...")
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)
    
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("No numeric features for VIF calculation.")
        return pd.DataFrame()
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_df.columns
    
    try:
        vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) for i in range(len(numeric_df.columns))]
        vif_data = vif_data.sort_values('VIF', ascending=False)
        
        vif_output_path = os.path.join(ARTIFACTS_DIR, 'vif_scores.csv')
        vif_data.to_csv(vif_output_path, index=False)
        
        print("VIF Scores (Top 10):")
        print(vif_data.head(10).to_string(index=False))
        
        return vif_data
    except Exception as e:
        print(f"VIF calculation failed: {e}")
        return pd.DataFrame() 