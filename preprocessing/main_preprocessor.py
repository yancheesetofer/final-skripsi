# FEEDBACK
# preprocess_pipeline/main_preprocess.py
# Pisahkan data real dan data artifisial
# kasih contoh output
# buat sesuai gambar
# judul subbab, bab dan tabel
# gambarnya disesuaikan atau tidak 
# mactexnogui
# terlalu teknis, buat lebih naratif, buat lebih mudah dipahami
# lebih intuitif. asumsi: orang baca langsung ngerti.
# buat lebih ringkas, buat lebih padat.
# ganti judul jadi metode penilitian bab 3
# dibagi dua gambarnya, pipeline tetap ada -> graf tahapan penelitian
# fokus saja, diremove agar 
# tabel tidak intuitif
# judul sama agar lebih cepat dipahami : tahapan penelitian + pipeline, ubah mudah dibaca
# bab 4 ganti eksperimen dan analisis
# dibalik masuk ke analisis, hasil deteksi dll, dampak perbedaan dataset dll
# SENIN HARD DEADLINE

import pandas as pd
import os
import json
from data_loader import load_all_data # Assuming EXPECTED_TABLES is defined in data_loader
from core_processor import clean_and_prepare_event_data
from feature_extractor import extract_intra_attempt_features, calculate_similarity_features, calculate_comparative_features
from feature_processor import (
    handle_missing_values, 
    scale_features, 
    select_features_by_collinearity, # New
    select_features_by_variance,   # New
    ARTIFACTS_DIR
)

def run_preprocessing_pipeline(data_input_dir, output_feature_path, train_mode=True, ground_truth_path=None):
    """
    Orchestrates the full preprocessing pipeline including improved feature selection.
    """
    print(f"Running preprocessing pipeline. Train mode: {train_mode}")
    print(f"Input data directory: {data_input_dir}")
    print(f"Output feature path: {output_feature_path}")

    # --- Step 1: Load Data ---
    raw_data_dict = load_all_data(data_input_dir)

    # --- Step 2: Core Preprocessing & Sessionization ---
    processed_event_log = clean_and_prepare_event_data(raw_data_dict)
    if processed_event_log.empty:
        print("Preprocessing stopped: Processed event log is empty.")
        return pd.DataFrame() # Return empty DataFrame

    # --- Step 3: Feature Engineering ---
    # Calculate questions per quiz dynamically from actual data
    q_per_quiz_map = {}
    
    # Method 1: Count from question_attempts linked to quiz_attempts
    if not raw_data_dict['mdl_quiz_attempts'].empty and not raw_data_dict['mdl_question_attempts'].empty:
        # Link quiz attempts to question attempts via question_usage_id
        quiz_attempts = raw_data_dict['mdl_quiz_attempts'][['quiz_id', 'question_usage_id']].copy()
        question_attempts = raw_data_dict['mdl_question_attempts'][['question_usage_id', 'questionid']].copy()
        
        # Count unique questions per quiz
        quiz_to_questions = pd.merge(quiz_attempts, question_attempts, on='question_usage_id', how='inner')
        if not quiz_to_questions.empty:
            q_counts = quiz_to_questions.groupby('quiz_id')['questionid'].nunique().to_dict()
            q_per_quiz_map.update(q_counts)
            print(f"Calculated questions per quiz from data: {dict(list(q_counts.items())[:5])}{'...' if len(q_counts) > 5 else ''}")
    
    # Fallback: Use config file if available, otherwise default
    if not q_per_quiz_map:
        gen_config_path = os.path.join(data_input_dir, 'generator_config.json')
        if os.path.exists(gen_config_path):
            with open(gen_config_path, 'r') as f_cfg:
                gen_cfg = json.load(f_cfg)
                q_val = gen_cfg.get("questions_per_quiz", 10)
                if not raw_data_dict['mdl_quiz'].empty:
                    for qid_ in raw_data_dict['mdl_quiz']['quiz_id'].unique(): 
                        q_per_quiz_map[qid_] = q_val
                    print(f"Used config questions_per_quiz: {q_val}")
        else:
            print("Warning: Could not determine questions_per_quiz from data or config. Using default=10.")
            if not raw_data_dict['mdl_quiz'].empty:
                for qid_ in raw_data_dict['mdl_quiz']['quiz_id'].unique(): 
                    q_per_quiz_map[qid_] = 10

    attempt_level_features = extract_intra_attempt_features(processed_event_log, q_per_quiz_map)
    if attempt_level_features.empty:
        print("Preprocessing stopped: No features extracted at attempt level.")
        return pd.DataFrame()
    
    attempt_level_features_sim = calculate_similarity_features(attempt_level_features.copy())
    
    available_cols = attempt_level_features_sim.columns
    cols_for_zscore_candidate = ['total_duration', 'num_actions', 'mean_step_duration', 
                                 'nav_revisits_count', 'nav_entropy', 'mean_nav_similarity', 
                                 'max_nav_similarity', 'std_nav_similarity', 'high_nav_similarity_count']
    actual_cols_for_zscore = [col for col in cols_for_zscore_candidate if col in available_cols]
    engineered_features = calculate_comparative_features(attempt_level_features_sim, actual_cols_for_zscore)

    if engineered_features.empty:
        print("Preprocessing stopped: No features after comparative calculation.")
        return pd.DataFrame()

    # --- Step 4: Feature Post-Processing (Imputation, Selection, Scaling) ---
    
    # CRITICAL FIX: Separate ID columns from features to prevent them from being processed
    id_cols = ['attempt_id', 'user_id', 'quiz_id']
    id_data = engineered_features[id_cols].copy()  # Preserve original ID values
    feature_data = engineered_features.drop(columns=id_cols, errors='ignore')
    
    # Add ground truth labels BEFORE feature processing to preserve original user_id mapping
    if train_mode and ground_truth_path:
        if os.path.exists(ground_truth_path):
            gt_df = pd.read_csv(ground_truth_path)
            user_gt_map = gt_df.set_index('user_id')['is_cheater'].to_dict()
            id_data['is_cheater'] = id_data['user_id'].map(user_gt_map).fillna(0).astype(int)
            
            user_group_map = gt_df.set_index('user_id')['cheating_group_id'].to_dict()
            id_data['cheating_group_id'] = id_data['user_id'].map(user_group_map).fillna("N/A")
            print(f"Added ground truth labels. Cheater count: {id_data['is_cheater'].sum()}")
        else:
            print(f"Warning: Ground truth file not found at {ground_truth_path}. Labels not added.")
            id_data['is_cheater'] = 0
            id_data['cheating_group_id'] = "N/A"
    
    # 4.1 Imputation (only on feature data)
    imputed_features = handle_missing_values(feature_data, strategy='mean', train_mode=train_mode)

    # 4.2 Collinearity Selection (only on feature data)
    collinearity_selected_features = select_features_by_collinearity(
        imputed_features, 
        vif_threshold=10.0,
        corr_threshold=0.95,
        train_mode=train_mode
    )
    
    # 4.3 Variance Threshold Selection (only on feature data)
    variance_selected_features = select_features_by_variance(
        collinearity_selected_features,
        threshold=0.01,
        train_mode=train_mode
    )
    
    # 4.4 Scaling (only on feature data)
    final_scaled_features = scale_features(
        variance_selected_features, 
        scaler_type='standard', 
        train_mode=train_mode
    )
    
    # 4.5 Recombine processed features with original ID data
    final_output_df = pd.concat([id_data.reset_index(drop=True), final_scaled_features.reset_index(drop=True)], axis=1)
    
    # Reorder columns to have IDs first, then features, then labels if present
    ordered_cols = []
    id_cols_ordered = ['attempt_id', 'user_id', 'quiz_id']
    label_cols_ordered = ['is_cheater', 'cheating_group_id']
    
    for col in id_cols_ordered:
        if col in final_output_df.columns: ordered_cols.append(col)
            
    feature_data_cols = [col for col in final_output_df.columns if col not in id_cols_ordered + label_cols_ordered]
    ordered_cols.extend(sorted(feature_data_cols)) # Sort feature columns for consistency

    for col in label_cols_ordered:
        if col in final_output_df.columns: ordered_cols.append(col)

    final_output_df = final_output_df[ordered_cols]


    try:
        os.makedirs(os.path.dirname(output_feature_path), exist_ok=True)
        final_output_df.to_csv(output_feature_path, index=False)
        print(f"Successfully saved final processed features ({final_output_df.shape[0]} rows, {final_output_df.shape[1]} cols) to: {output_feature_path}")
    except Exception as e:
        print(f"Error saving processed features: {e}")

    print("Preprocessing pipeline finished.")
    return final_output_df


if __name__ == "__main__":
    print("\n" + "="*20 + " PROCESSING ARTIFICIAL DATA (TRAIN MODE) " + "="*20)
    artificial_data_dir = '../data/moodle_logs_final_viz_corrected/'
    artificial_gt_path = os.path.join(artificial_data_dir, 'cheating_ground_truth.csv')
    artificial_output_features = '../data/processed_artificial_features_V2.csv'
    
    if os.path.exists(ARTIFACTS_DIR):
        for f_artifact in os.listdir(ARTIFACTS_DIR):
            if f_artifact.endswith((".joblib", ".json")):
                try:
                    os.remove(os.path.join(ARTIFACTS_DIR, f_artifact))
                    print(f"Removed old artifact: {f_artifact}")
                except OSError as e:
                    print(f"Error removing artifact {f_artifact}: {e}")

    final_df = run_preprocessing_pipeline(
        data_input_dir=artificial_data_dir,
        output_feature_path=artificial_output_features,
        train_mode=True,
        ground_truth_path=artificial_gt_path
    )
    if not final_df.empty:
        print("\nSample of final processed data for training:")
        print(final_df.head())
        print(f"Final selected features for training: {len([c for c in final_df.columns if c not in ['attempt_id', 'user_id', 'quiz_id', 'is_cheater', 'cheating_group_id']])}")

    # Real data processing (Detection Mode)
    print("\n" + "="*20 + " PROCESSING REAL DATA (DETECTION MODE) " + "="*20)
    real_data_dir = '../institute_log_legacy/lumbung_sampled/'  # Real-world data from institute
    real_output_features = '../data/processed_real_features_for_detection_V2.csv'
    
    if os.path.exists(real_data_dir):
        real_df = run_preprocessing_pipeline(
            data_input_dir=real_data_dir,
            output_feature_path=real_output_features,
            train_mode=False,  # Detection mode - uses saved artifacts from training
            ground_truth_path=None  # No ground truth for real data
        )
        if not real_df.empty:
            print("\nSample of final processed real data for detection:")
            print(real_df.head())
            print(f"Final features for detection: {len([c for c in real_df.columns if c not in ['attempt_id', 'user_id', 'quiz_id', 'is_cheater', 'cheating_group_id']])}")
            print(f"Total real-world attempts processed: {len(real_df)}")
        else:
            print("Warning: No real data was processed.")
    else:
        print(f"Warning: Real data directory not found at {real_data_dir}")
        print("Skipping real data processing.")