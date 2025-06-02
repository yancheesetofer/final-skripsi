# preprocess_pipeline/core_preprocessor.py
import pandas as pd
import numpy as np

# --- Thesis Section 3.4.1: Pembersihan Data (Data Cleaning) ---
# --- Thesis Section 3.4.2: Transformasi dan Normalisasi Data ---

def unify_timestamps(df, column_names):
    """Converts specified timestamp columns to POSIX seconds (integers)."""
    for col in column_names:
        if col in df.columns:
            # Assuming timestamps are already POSIX or can be directly converted
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            # If they were datetime objects from loader: df[col] = df[col].astype(np.int64) // 10**9
    return df

def clean_and_prepare_event_data(loaded_data):
    """
    Cleans individual log tables and merges them to create a base DataFrame
    for quiz attempt analysis.
    """
    print("Starting core preprocessing of event data...")

    # Timestamp unification
    ts_cols_quiz_attempts = ['timestart', 'timefinish']
    loaded_data['mdl_quiz_attempts'] = unify_timestamps(loaded_data['mdl_quiz_attempts'], ts_cols_quiz_attempts)
    
    ts_cols_steps = ['timecreated']
    loaded_data['mdl_question_attempt_steps'] = unify_timestamps(loaded_data['mdl_question_attempt_steps'], ts_cols_steps)

    # Filter unfinished attempts if necessary (as per thesis 3.3.1.1 for real data)
    # For generated data, 'state' is usually 'finished'
    # Example: attempts_df = loaded_data['mdl_quiz_attempts'][loaded_data['mdl_quiz_attempts']['state'] == 'finished']
    attempts_df = loaded_data['mdl_quiz_attempts'].copy()
    if attempts_df.empty:
        print("Warning: No 'finished' attempts found or attempts_df is empty after filtering.")
        return pd.DataFrame()

    # Merge attempt steps with step data to get answer values
    steps_df = loaded_data['mdl_question_attempt_steps']
    step_data_df = loaded_data['mdl_question_attempt_step_data']
    
    # Filter for 'answer' events in step_data and rename value to 'answer_value'
    answer_data_df = step_data_df[step_data_df['name'] == 'answer'].rename(columns={'value': 'answer_value'})
    
    if not steps_df.empty and not answer_data_df.empty:
        steps_detailed_df = pd.merge(steps_df, answer_data_df[['question_step_id', 'answer_value']], 
                                     on='question_step_id', how='left')
    elif not steps_df.empty:
        steps_detailed_df = steps_df.copy()
        steps_detailed_df['answer_value'] = pd.NA
    else:
        steps_detailed_df = pd.DataFrame()

    # Merge with question-specific attempts to get questionid and maxmark
    # mdl_question_attempts is the question-specific log from generate_case.py
    question_attempts_real_df = loaded_data['mdl_question_attempts']
    if not steps_detailed_df.empty and not question_attempts_real_df.empty:
        steps_with_questions_df = pd.merge(steps_detailed_df, 
                                           question_attempts_real_df[['question_attempt_id', 'questionid', 'maxmark', 'question_usage_id']],
                                           on='question_attempt_id', how='left')
    else:
        steps_with_questions_df = steps_detailed_df # Or empty if steps_detailed_df is empty

    # Merge this detailed step log with the main attempts_df
    # Need common key: 'question_usage_id' connects mdl_quiz_attempts to mdl_question_attempts (which links to steps)
    # However, generate_case.py structure implies mdl_question_attempts links directly to mdl_question_attempt_steps via question_attempt_id
    # And mdl_quiz_attempts links to mdl_question_usages (question_usage_id)
    # For the generated data, let's link attempts_df to steps_with_questions_df via user_id and quiz_id
    # A more robust approach for real Moodle logs involves careful joining based on attempt IDs and usage IDs.

    # Simplified join for generated data structure:
    # Each quiz_attempt (attempt_id) has one question_usage_id
    # Each question_attempt_real (question_attempt_id) links to one question_usage_id
    # So, we can try to link through question_usage_id if it's consistent.
    
    # Let's try to reconstruct events per quiz_attempt_id from the main attempts_df
    # For each row in attempts_df (which is unique by attempt_id), we need to find its steps.
    # The `question_usage_id` on `attempts_df` is the key to `question_attempts_real_df` (which contains `question_attempt_id`)
    
    if 'question_usage_id' not in question_attempts_real_df.columns and not question_attempts_real_df.empty:
         print("Warning: 'question_usage_id' missing in 'mdl_question_attempts'. Cannot directly link quiz attempts to question details robustly.")
         # Fallback strategy or error
    
    # Merge attempts with question usage details (which contains question_attempt_id)
    # This chain is: mdl_quiz_attempts -> mdl_question_usages (implicit, by question_usage_id) -> mdl_question_attempts (question_attempt_id) -> mdl_question_attempt_steps
    
    # For the provided structure, steps_with_questions_df already contains 'question_attempt_id' and 'questionid'
    # We need to associate these steps back to the main 'attempt_id' from 'mdl_quiz_attempts'.
    # The 'question_usage_id' links 'mdl_quiz_attempts' to 'mdl_question_attempts' (the question-specific one, here called `question_attempts_real_df`)

    # 1. Link quiz_attempts to question_attempts_real via question_usage_id
    if not attempts_df.empty and not question_attempts_real_df.empty:
        # Ensure 'question_usage_id' is in both for the merge
        if 'question_usage_id' in attempts_df.columns and 'question_usage_id' in question_attempts_real_df.columns:
            merged_attempts_q_attempts = pd.merge(
                attempts_df[['attempt_id', 'quiz_id', 'user_id', 'timestart', 'timefinish', 'sumgrades', 'question_usage_id']],
                question_attempts_real_df[['question_attempt_id', 'question_usage_id', 'questionid', 'maxmark']],
                on='question_usage_id',
                how='left'
            )
        else:
            print("Warning: 'question_usage_id' missing from attempts_df or question_attempts_real_df. Merging might be incomplete.")
            merged_attempts_q_attempts = attempts_df[['attempt_id', 'quiz_id', 'user_id', 'timestart', 'timefinish', 'sumgrades', 'question_usage_id']].copy()
            # Add missing columns that would come from question_attempts_real_df
            for col in ['question_attempt_id', 'questionid', 'maxmark']:
                 if col not in merged_attempts_q_attempts.columns: merged_attempts_q_attempts[col] = pd.NA


        # 2. Link this to detailed steps
        if not merged_attempts_q_attempts.empty and not steps_with_questions_df.empty:
             # Ensure 'question_attempt_id' is present for merging with steps_with_questions_df (which already has questionid)
            if 'question_attempt_id' in merged_attempts_q_attempts.columns and 'question_attempt_id' in steps_with_questions_df.columns:
                # We might have duplicate 'questionid' and 'maxmark' if they were in steps_with_questions_df already.
                # Let's ensure steps_with_questions_df provides the core step info.
                cols_from_steps = ['question_attempt_id', 'question_step_id', 'sequencenumber', 'state', 'timecreated', 'answer_value']
                # Add questionid from steps_with_questions_df if not already properly merged
                if 'questionid' not in cols_from_steps and 'questionid' in steps_with_questions_df.columns:
                    cols_from_steps.append('questionid')

                final_event_log = pd.merge(
                    merged_attempts_q_attempts,
                    steps_with_questions_df[cols_from_steps], # Select specific columns from steps
                    on=['question_attempt_id', 'questionid'] if 'questionid' in merged_attempts_q_attempts.columns and 'questionid' in steps_with_questions_df.columns else 'question_attempt_id',
                    how='left'
                )
            else:
                 print("Warning: 'question_attempt_id' missing for merging attempts with steps. Final event log may be incomplete.")
                 final_event_log = merged_attempts_q_attempts
                 for col in ['question_step_id', 'sequencenumber', 'state', 'timecreated', 'answer_value']:
                     if col not in final_event_log.columns: final_event_log[col] = pd.NA
        else:
            final_event_log = merged_attempts_q_attempts # Or empty if merged_attempts_q_attempts is empty
    else:
        final_event_log = pd.DataFrame() # Return empty if initial attempts_df is empty

    if not final_event_log.empty:
        # Sort events chronologically within each attempt
        final_event_log = final_event_log.sort_values(by=['attempt_id', 'timecreated', 'sequencenumber'])
        print(f"Core preprocessing complete. Final event log has {final_event_log.shape[0]} events.")
    else:
        print("Warning: Final event log is empty after core preprocessing.")
        
    return final_event_log


if __name__ == '__main__':
    data_dir = 'data/moodle_logs_final_viz_corrected/' 
    raw_data = load_all_data(data_dir)
    processed_event_log = clean_and_prepare_event_data(raw_data)
    
    # if not processed_event_log.empty:
    #     print("\nProcessed Event Log Sample:")
    #     print(processed_event_log.head())
    #     print(f"\nUnique attempts in processed log: {processed_event_log['attempt_id'].nunique()}")
    #     # Further checks, e.g., how many attempts have step data
    #     # print(processed_event_log.groupby('attempt_id')['question_step_id'].count().describe())