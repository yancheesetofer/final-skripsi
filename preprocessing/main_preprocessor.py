# preprocess_pipeline/main_preprocess.py
import pandas as pd
import os
import json
from data_loader import load_all_data, EXPECTED_TABLES
from core_processor import clean_and_prepare_event_data
from feature_extractor import extract_intra_attempt_features, calculate_similarity_features, calculate_comparative_features
from feature_processor import handle_missing_values, scale_features, calculate_vif, ARTIFACTS_DIR

def run_preprocessing_pipeline(data_input_dir, output_feature_path, train_mode=True, ground_truth_path=None):
    """
    Orchestrates the full preprocessing pipeline.
    - data_input_dir: Directory containing raw CSV log files.
    - output_feature_path: Path to save the final processed feature DataFrame.
    - train_mode: Boolean. If True, fits and saves imputers/scalers. If False, loads and applies them.
    - ground_truth_path: Path to cheating_ground_truth.csv (used for attaching labels if in train_mode).
    """
    print(f"Running preprocessing pipeline. Train mode: {train_mode}")
    print(f"Input data directory: {data_input_dir}")
    print(f"Output feature path: {output_feature_path}")

    # --- Step 1: Load Data ---
    # # For generated data from your structure:
    # # data_input_dir = 'data/moodle_logs_final_viz_corrected/'
    raw_data_dict = load_all_data(data_input_dir)

    # --- Step 2: Core Preprocessing & Sessionization ---
    processed_event_log = clean_and_prepare_event_data(raw_data_dict)
    if processed_event_log.empty:
        print("Preprocessing stopped: Processed event log is empty.")
        return

    # --- Step 3: Feature Engineering ---
    # Determine questions_per_quiz (important for some sequential features like linearity)
    # This logic should be robust for both artificial and real data.
    q_per_quiz_map = {}
    gen_config_path = os.path.join(data_input_dir, 'generator_config.json') # Specific to generated data
    if os.path.exists(gen_config_path):
        with open(gen_config_path, 'r') as f_cfg:
            gen_cfg = json.load(f_cfg)
            q_val = gen_cfg.get("questions_per_quiz", 10)
            if not raw_data_dict['mdl_quiz'].empty:
                for qid_ in raw_data_dict['mdl_quiz']['quiz_id'].unique(): q_per_quiz_map[qid_] = q_val
    else: # Fallback for real data: try to infer from mdl_Youtubes or mdl_question_attempts per quiz
        if not raw_data_dict['mdl_Youtubes'].empty and not raw_data_dict['mdl_quiz'].empty:
            # This requires joining questions to quizzes
            # Placeholder: assume a default or a more complex lookup for real data
            print("Warning: generator_config.json not found. questions_per_quiz might be inaccurate for real data.")
            for qid_ in raw_data_dict['mdl_quiz']['quiz_id'].unique(): q_per_quiz_map[qid_] = 10 # Default

    attempt_level_features = extract_intra_attempt_features(processed_event_log, q_per_quiz_map)
    if attempt_level_features.empty:
        print("Preprocessing stopped: No features extracted at attempt level.")
        return
    
    attempt_level_features_sim = calculate_similarity_features(attempt_level_features.copy()) # Pass copy
    
    # Define columns for Z-score calculation based on available features
    # These should be features that make sense to compare relative to a quiz's average
    available_cols = attempt_level_features_sim.columns
    cols_for_zscore_candidate = ['total_duration', 'num_actions', 'mean_step_duration', 
                                 'nav_revisits_count', 'nav_entropy', 'mean_nav_similarity', 
                                 'max_nav_similarity', 'std_nav_similarity', 'high_nav_similarity_count']
    
    actual_cols_for_zscore = [col for col in cols_for_zscore_candidate if col in available_cols]
    
    engineered_features = calculate_comparative_features(attempt_level_features_sim, actual_cols_for_zscore)
    if engineered_features.empty:
        print("Preprocessing stopped: No features after comparative calculation.")
        return

    # --- Step 4: Feature Post-Processing ---
    # Select only feature columns for imputation and scaling (exclude IDs for now)
    id_cols = ['attempt_id', 'user_id', 'quiz_id']
    feature_cols_to_process = [col for col in engineered_features.columns if col not in id_cols]
    features_to_process_df = engineered_features[feature_cols_to_process]

    # Handle missing values that might have been introduced
    # PROFESSOR_REVIEW: Justify imputation strategy. 'mean' is a default.
    # For consistency, the imputer path must be the same for train and test.
    imputer_path = os.path.join(ARTIFACTS_DIR, 'feature_imputer.joblib')
    processed_features_imputed = handle_missing_values(features_to_process_df, strategy='mean', 
                                                       train_mode=train_mode, imputer_path=imputer_path)

    # Scale features
    # PROFESSOR_REVIEW: Justify scaler choice. 'standard' is a common default.
    scaler_path = os.path.join(ARTIFACTS_DIR, 'standard_scaler.joblib')
    scaled_features_df = scale_features(processed_features_imputed, scaler_type='standard', 
                                        train_mode=train_mode, scaler_path=scaler_path)

    # Add back ID columns
    for id_col in id_cols:
        if id_col in engineered_features.columns:
            scaled_features_df[id_col] = engineered_features[id_col]

    # Optional: VIF calculation (for analysis, usually done on training data)
    if train_mode:
        _ = calculate_vif(scaled_features_df.drop(columns=id_cols, errors='ignore'))
    
    # Add ground truth labels if in training mode and path provided
    if train_mode and ground_truth_path:
        if os.path.exists(ground_truth_path):
            gt_df = pd.read_csv(ground_truth_path)
            # Merge based on user_id. This assumes one label per user. 
            # If labels are per attempt, adjust merge keys.
            # For this project, ground truth is per user. Model will predict per attempt, then aggregate.
            user_gt_map = gt_df.set_index('user_id')['is_cheater'].to_dict()
            scaled_features_df['is_cheater'] = scaled_features_df['user_id'].map(user_gt_map)
            # Handle users not in ground_truth (e.g., if using partial real data for training)
            # For fully artificial data, all users should be in ground_truth.
            scaled_features_df['is_cheater'] = scaled_features_df['is_cheater'].fillna(0).astype(int) 
            
            user_group_map = gt_df.set_index('user_id')['cheating_group_id'].to_dict()
            scaled_features_df['cheating_group_id'] = scaled_features_df['user_id'].map(user_group_map)
            scaled_features_df['cheating_group_id'] = scaled_features_df['cheating_group_id'].fillna("N/A")
            
            print(f"Added 'is_cheater' and 'cheating_group_id' labels. Cheater count: {scaled_features_df['is_cheater'].sum()}")
        else:
            print(f"Warning: Ground truth file not found at {ground_truth_path}. Labels not added.")

    # Save the final processed features
    try:
        os.makedirs(os.path.dirname(output_feature_path), exist_ok=True)
        scaled_features_df.to_csv(output_feature_path, index=False)
        print(f"Successfully saved processed features to: {output_feature_path}")
    except Exception as e:
        print(f"Error saving processed features: {e}")

    print("Preprocessing pipeline finished.")


if __name__ == "__main__":
    # --- For processing ARTIFICIAL data (TRAIN MODE) ---
    print("\n" + "="*20 + " PROCESSING ARTIFICIAL DATA (TRAIN MODE) " + "="*20)
    artificial_data_dir = '../data/moodle_logs_final_viz_corrected/' # Your artificial data
    artificial_gt_path = os.path.join(artificial_data_dir, 'cheating_ground_truth.csv')
    artificial_output_features = '../data/processed_artificial_features.csv' # Output for training
    
    # Ensure artifacts directory is clean or managed if re-running training
    if os.path.exists(ARTIFACTS_DIR):
        for f_artifact in os.listdir(ARTIFACTS_DIR):
            if f_artifact.endswith(".joblib") or f_artifact.endswith(".csv"): # Clear previous VIF too
                try:
                    os.remove(os.path.join(ARTIFACTS_DIR, f_artifact))
                    print(f"Removed old artifact: {f_artifact}")
                except OSError as e:
                    print(f"Error removing artifact {f_artifact}: {e}")


    run_preprocessing_pipeline(
        data_input_dir=artificial_data_dir,
        output_feature_path=artificial_output_features,
        train_mode=True,
        ground_truth_path=artificial_gt_path
    )

    # --- For processing REAL data (DETECTION MODE) ---
    # This is a placeholder. You'd replace 'path_to_real_data_dir' with the actual path.
    # print("\n" + "="*20 + " PROCESSING REAL DATA (DETECTION MODE) " + "="*20)
    # real_data_dir = 'path_to_real_data_dir/' 
    # real_output_features = 'data/processed_real_features_for_detection.csv'
    
    # if os.path.exists(real_data_dir): # Only run if real data path exists
    #     run_preprocessing_pipeline(
    #         data_input_dir=real_data_dir,
    #         output_feature_path=real_output_features,
    #         train_mode=False, # IMPORTANT: Use False for real/test data
    #         ground_truth_path=None # No GT for real detection data usually
    #     )
    # else:
    #     print(f"Real data directory '{real_data_dir}' not found. Skipping real data processing example.")