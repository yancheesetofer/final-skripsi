# preprocess_pipeline/feature_extractor.py
import pandas as pd
import numpy as np
from scipy.stats import entropy
from Levenshtein import distance as levenshtein_distance # Python-Levenshtein package
# from dtw import dtw # dtw-python package for dynamic time warping
# For DTW, you might need to install it: pip install dtw-python

# --- Thesis Section 3.6: Feature Engineering ---
# --- Thesis Section 3.4.3: Ekstraksi Fitur dan Deteksi Outlier ---

def extract_intra_attempt_features(processed_event_log_df, questions_per_quiz):
    """Extracts features for each user-quiz attempt."""
    if processed_event_log_df.empty:
        return pd.DataFrame()

    print("Extracting intra-attempt features...")
    
    # Group by attempt_id to process each attempt
    # Ensure 'questionid' is present and correctly typed for sequence analysis
    if 'questionid' in processed_event_log_df.columns:
        processed_event_log_df['questionid'] = pd.to_numeric(processed_event_log_df['questionid'], errors='coerce')
    else:
        print("Warning: 'questionid' column missing in processed_event_log_df. Sequential features will be affected.")
        # Add a dummy questionid if missing to avoid crashing, though features will be meaningless
        processed_event_log_df['questionid'] = 0 

    # Ensure 'timecreated' is present for timing features
    if 'timecreated' not in processed_event_log_df.columns:
         print("Warning: 'timecreated' column missing. Timing features will be affected.")
         processed_event_log_df['timecreated'] = 0 # Dummy for calculation

    # Calculate duration of each step (time between consecutive actions by the same user in the same attempt)
    processed_event_log_df = processed_event_log_df.sort_values(by=['attempt_id', 'timecreated', 'sequencenumber'])
    processed_event_log_df['step_duration'] = processed_event_log_df.groupby('attempt_id')['timecreated'].diff().fillna(0)

    features_list = []
    for attempt_id, group in processed_event_log_df.groupby('attempt_id'):
        attempt_features = {}
        # Basic info
        attempt_features['attempt_id'] = attempt_id
        attempt_features['user_id'] = group['user_id'].iloc[0]
        attempt_features['quiz_id'] = group['quiz_id'].iloc[0]
        
        # --- Thesis Section 3.6.1: Ekstraksi Fitur Dasar ---
        # Timing features
        attempt_features['total_duration'] = (group['timefinish'].iloc[0] - group['timestart'].iloc[0]) if 'timefinish' in group.columns and 'timestart' in group.columns else 0
        attempt_features['num_actions'] = group.shape[0]
        
        step_durations = group['step_duration'][group['step_duration'] > 0] # Exclude initial 0
        if not step_durations.empty:
            attempt_features['mean_step_duration'] = step_durations.mean()
            attempt_features['std_step_duration'] = step_durations.std(ddof=0) # Use ddof=0 for population std if appropriate
            attempt_features['median_step_duration'] = step_durations.median()
            attempt_features['quick_actions_count'] = (step_durations < 5).sum() # e.g., less than 5 seconds
            attempt_features['long_actions_count'] = (step_durations > 600).sum() # e.g., more than 10 minutes
        else:
            attempt_features['mean_step_duration'] = 0
            attempt_features['std_step_duration'] = 0
            attempt_features['median_step_duration'] = 0
            attempt_features['quick_actions_count'] = 0
            attempt_features['long_actions_count'] = 0

        # Performance
        attempt_features['sumgrades'] = group['sumgrades'].iloc[0] if 'sumgrades' in group.columns else 0
        
        # --- Thesis Section 3.6.2: Ekstraksi Fitur Sequence ---
        nav_sequence = group['questionid'].dropna().astype(int).tolist() # Sequence of question IDs visited
        if nav_sequence:
            attempt_features['nav_seq_length'] = len(nav_sequence)
            attempt_features['nav_unique_questions_visited'] = len(set(nav_sequence))
            
            # Revisits (count of questions visited more than once)
            q_counts = pd.Series(nav_sequence).value_counts()
            attempt_features['nav_revisits_count'] = (q_counts > 1).sum()
            attempt_features['nav_total_revisits'] = (q_counts[q_counts > 1] - 1).sum()

            # Linearity (simple version: ratio of unique questions to ideal minimum steps)
            # A more sophisticated measure could compare to the actual question order
            ideal_min_steps = questions_per_quiz.get(attempt_features['quiz_id'], attempt_features['nav_unique_questions_visited'])
            if ideal_min_steps == 0: ideal_min_steps = 1 # Avoid division by zero
            attempt_features['nav_linearity'] = attempt_features['nav_unique_questions_visited'] / ideal_min_steps if attempt_features['nav_unique_questions_visited'] > 0 else 0
            
            # Entropy of navigation sequence
            if attempt_features['nav_unique_questions_visited'] > 0:
                counts = np.bincount(nav_sequence)
                probs = counts[counts > 0] / len(nav_sequence)
                attempt_features['nav_entropy'] = entropy(probs, base=2)
            else:
                attempt_features['nav_entropy'] = 0
        else:
            attempt_features['nav_seq_length'] = 0
            attempt_features['nav_unique_questions_visited'] = 0
            attempt_features['nav_revisits_count'] = 0
            attempt_features['nav_total_revisits'] = 0
            attempt_features['nav_linearity'] = 0
            attempt_features['nav_entropy'] = 0
            
        # Store the raw navigation sequence for similarity calculation later
        attempt_features['raw_nav_sequence'] = nav_sequence
        
        features_list.append(attempt_features)
        
    attempt_features_df = pd.DataFrame(features_list)
    print(f"Intra-attempt feature extraction complete. Generated {attempt_features_df.shape[0]} attempt feature rows.")
    return attempt_features_df

def calculate_similarity_features(attempt_features_df):
    """Calculates inter-user similarity features for each quiz."""
    # --- Thesis Section 3.6.3: Perhitungan Similarity Features ---
    if attempt_features_df.empty or 'raw_nav_sequence' not in attempt_features_df.columns:
        print("Warning: Cannot calculate similarity features. DataFrame is empty or 'raw_nav_sequence' is missing.")
        # Add empty similarity columns if df is not empty but sequence is missing
        if not attempt_features_df.empty:
            for sim_type in ['nav', 'answer', 'timing']: # Example types
                 attempt_features_df[f'mean_{sim_type}_similarity'] = 0.0
                 attempt_features_df[f'max_{sim_type}_similarity'] = 0.0
        return attempt_features_df
    
    print("Calculating similarity features...")
    all_similarity_features = []

    for quiz_id, quiz_group_df in attempt_features_df.groupby('quiz_id'):
        user_attempts_in_quiz = quiz_group_df.to_dict('records')
        num_users_in_quiz = len(user_attempts_in_quiz)
        
        if num_users_in_quiz <= 1: # Need at least two users to compare
            for user_attempt in user_attempts_in_quiz:
                sim_features = {'attempt_id': user_attempt['attempt_id']}
                sim_features['mean_nav_similarity'] = 0.0
                sim_features['max_nav_similarity'] = 0.0
                # Add others as 0 too
                all_similarity_features.append(sim_features)
            continue

        # Initialize similarity matrices for this quiz (user_idx_in_quiz x user_idx_in_quiz)
        nav_sim_matrix = np.zeros((num_users_in_quiz, num_users_in_quiz))
        # Placeholder for timing and answer similarity matrices

        for i in range(num_users_in_quiz):
            for j in range(i + 1, num_users_in_quiz):
                seq1 = user_attempts_in_quiz[i]['raw_nav_sequence']
                seq2 = user_attempts_in_quiz[j]['raw_nav_sequence']
                
                # Navigation Similarity (Levenshtein)
                # Normalize by the length of the longer sequence
                if seq1 and seq2: # Both sequences must be non-empty
                    max_len = max(len(seq1), len(seq2))
                    if max_len > 0:
                        # PROFESSOR_REVIEW: Justification for Levenshtein distance and normalization method.
                        # Levenshtein distance measures edit distance. Normalizing makes it comparable.
                        # Similarity = 1 - (Normalized Edit Distance)
                        lev_dist = levenshtein_distance("".join(map(str, seq1)), "".join(map(str, seq2))) # Treat q_ids as chars
                        nav_sim_matrix[i, j] = nav_sim_matrix[j, i] = 1.0 - (lev_dist / max_len)
                    else: # Both sequences empty
                        nav_sim_matrix[i, j] = nav_sim_matrix[j, i] = 1.0 
                elif not seq1 and not seq2: # Both empty
                     nav_sim_matrix[i, j] = nav_sim_matrix[j, i] = 1.0 # Or 0.0 depending on interpretation
                else: # One is empty, one is not
                     nav_sim_matrix[i, j] = nav_sim_matrix[j, i] = 0.0

                # TODO: Calculate and fill timing_sim_matrix, answer_sim_matrix
                # For example: Timing - Pearson correlation of step_durations sequences
                # For example: Answer - Jaccard index of common correct/incorrect answers

        # Aggregate similarity scores for each user
        for i in range(num_users_in_quiz):
            user_attempt = user_attempts_in_quiz[i]
            sim_features = {'attempt_id': user_attempt['attempt_id']}
            
            # Nav similarity aggregates
            user_nav_sims = np.concatenate((nav_sim_matrix[i, :i], nav_sim_matrix[i, i+1:])) # Exclude self
            if user_nav_sims.size > 0:
                sim_features['mean_nav_similarity'] = np.mean(user_nav_sims)
                sim_features['max_nav_similarity'] = np.max(user_nav_sims)
                sim_features['std_nav_similarity'] = np.std(user_nav_sims)
                 # PROFESSOR_REVIEW: Threshold for 'high similarity' (e.g., 0.7 or 0.8) should be justified.
                sim_features['high_nav_similarity_count'] = np.sum(user_nav_sims > 0.8) 
            else:
                sim_features['mean_nav_similarity'] = 0.0
                sim_features['max_nav_similarity'] = 0.0
                sim_features['std_nav_similarity'] = 0.0
                sim_features['high_nav_similarity_count'] = 0
            
            # TODO: Add aggregates for timing and answer similarities
            
            all_similarity_features.append(sim_features)

    similarity_features_df = pd.DataFrame(all_similarity_features)
    
    if not attempt_features_df.empty and not similarity_features_df.empty:
        attempt_features_df = pd.merge(attempt_features_df, similarity_features_df, on='attempt_id', how='left')
    elif not attempt_features_df.empty: # similarity_features_df might be empty if only 1 user per quiz
        for col_prefix in ['mean_nav', 'max_nav', 'std_nav', 'high_nav_similarity_count']: # Add more if other sims are added
            attempt_features_df[f'{col_prefix}_similarity' if 'count' not in col_prefix else col_prefix] = 0.0


    # Remove raw sequences after use
    if 'raw_nav_sequence' in attempt_features_df.columns:
        attempt_features_df = attempt_features_df.drop(columns=['raw_nav_sequence'])
        
    print(f"Similarity feature calculation complete.")
    return attempt_features_df

def calculate_comparative_features(attempt_features_df, feature_cols_for_comparison):
    """Calculates Z-scores for specified features, comparing user to quiz average."""
    # --- This corresponds to enhanced_model.py's create_quiz_specific_features ---
    if attempt_features_df.empty:
        return attempt_features_df
        
    print("Calculating comparative (Z-score) features...")
    
    # feature_cols_for_comparison = ['total_duration', 'num_actions', 'mean_step_duration', 'nav_revisits_count', 'nav_entropy'] # Example
    
    # Fill NaNs in columns to be used for Z-score calculation before grouping
    for col in feature_cols_for_comparison:
        if col in attempt_features_df.columns:
            attempt_features_df[col] = attempt_features_df[col].fillna(attempt_features_df[col].mean()) # Impute with overall mean first
        else:
            print(f"Warning: Column {col} for Z-score not found in DataFrame.")
            attempt_features_df[col] = 0 # Add as zero if missing to avoid errors

    # Define a function to calculate z-scores within each group
    def z_score_group(group, cols):
        for col in cols:
            if col in group.columns:
                mean = group[col].mean()
                std = group[col].std(ddof=0) # Use population std
                if std > 0: # Avoid division by zero
                    group[f'{col}_zscore'] = (group[col] - mean) / std
                else:
                    group[f'{col}_zscore'] = 0.0 # Or NaN, depending on how you want to treat no variance
        return group

    # Apply the z-score calculation grouped by quiz_id
    attempt_features_df = attempt_features_df.groupby('quiz_id', group_keys=False).apply(lambda x: z_score_group(x, feature_cols_for_comparison))
    
    print(f"Comparative (Z-score) feature calculation complete.")
    return attempt_features_df


if __name__ == '__main__':
    # Example usage:
    data_dir = 'data/moodle_logs_final_viz_corrected/' 
    raw_data = load_all_data(data_dir) # From data_loader.py
    processed_event_log = clean_and_prepare_event_data(raw_data) # From core_preprocessor.py
    
    # Get questions_per_quiz for linearity calculation
    # This would ideally come from mdl_quiz or by counting unique questions per quiz from logs
    q_per_quiz = {}
    if not raw_data['mdl_quiz'].empty and not raw_data['mdl_question_attempts'].empty:
        quiz_questions = raw_data['mdl_question_attempts'].groupby('question_usage_id')['questionid'].nunique()
        # Need to map question_usage_id to quiz_id
        usage_to_quiz = pd.Series(raw_data['mdl_quiz_attempts']['quiz_id'].values, index=raw_data['mdl_quiz_attempts']['question_usage_id']).to_dict()
        q_per_quiz = {usage_to_quiz.get(usage_id): count for usage_id, count in quiz_questions.items() if usage_to_quiz.get(usage_id) is not None}
        # Fallback if generate_case.py structure for questions_per_quiz config is available
        if not q_per_quiz:
             # Assuming generate_case.py uses a fixed number for all quizzes if not found
             config_path = os.path.join(data_dir, '../generator_config.json') # Path to generator_config.json
             if os.path.exists(config_path):
                with open(config_path, 'r') as f_cfg:
                    gen_cfg = json.load(f_cfg)
                    q_per_quiz_val = gen_cfg.get("questions_per_quiz", 10) # Default to 10
                    for qid_ in raw_data['mdl_quiz']['quiz_id'].unique(): q_per_quiz[qid_] = q_per_quiz_val


    attempt_level_features = extract_intra_attempt_features(processed_event_log, q_per_quiz)
    
    if not attempt_level_features.empty:
        attempt_level_features_with_sim = calculate_similarity_features(attempt_level_features.copy()) # Pass a copy
        
        cols_for_zscore = ['total_duration', 'num_actions', 'mean_step_duration', 
                           'nav_revisits_count', 'nav_entropy', 'mean_nav_similarity', 'max_nav_similarity'] # Example list
        # Filter out cols not present in df
        cols_for_zscore = [col for col in cols_for_zscore if col in attempt_level_features_with_sim.columns]

        final_features = calculate_comparative_features(attempt_level_features_with_sim, cols_for_zscore)
        
        # print("\nSample of Final Features:")
        # print(final_features.head())
        # final_features.info()