# preprocess_pipeline/data_loader.py
import pandas as pd
import os
import json

# --- Thesis Section 3.3: Sumber Data Log ---

EXPECTED_TABLES = {
    "mdl_user": {"user_id": int, "username": str}, # Add other relevant columns and types
    "mdl_quiz": {"quiz_id": int, "course": int, "quiz_name": str, "timeopen": int, "timeclose": int, "timelimit": int},
    "mdl_quiz_attempts": {"attempt_id": int, "quiz_id": int, "user_id": int, "timestart": int, "timefinish": int, "sumgrades": float, "state": str, "question_usage_id": int},
    "mdl_question_usages": {"question_usage_id": int, "context_id": int},
    "mdl_question_attempts": {"question_attempt_id": int, "question_usage_id": int, "questionid": int, "maxmark": float}, # This is the question-specific attempts log
    "mdl_question_attempt_steps": {"question_step_id": int, "question_attempt_id": int, "sequencenumber": int, "state": str, "timecreated": int},
    "mdl_question_attempt_step_data": {"step_data_id": int, "question_step_id": int, "name": str, "value": str},
    "mdl_Youtubes": {"Youtubes_id": int, "questionid": int, "answer_text": str, "fraction": float},
    "mdl_sessions": {"session_id": int, "user_id": int, "timecreated": int, "lastip": str}, # Simplified
    "mdl_quiz_grades": {"quiz_grades_id": int, "quiz_id": int, "user_id": int, "final_grade": float},
    "cheating_ground_truth": {"user_id": int, "is_cheater": int, "cheating_group_id": str, "cheating_severity": str} # For training labels
}

def load_csv_table(data_dir, table_name, schema):
    """Loads a single CSV table with basic validation and type casting."""
    file_path = os.path.join(data_dir, f"{table_name}.csv")
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found. Returning empty DataFrame.")
        return pd.DataFrame(columns=schema.keys())
    
    try:
        # Specify dtype for all columns to prevent misinterpretation, allow missing for non-critical
        # For simplicity here, we load then cast. For robustness, specify dtype in read_csv.
        df = pd.read_csv(file_path)
        
        # Validate columns
        for col, col_type in schema.items():
            if col not in df.columns:
                print(f"Warning: Expected column '{col}' not found in {table_name}.csv. Adding as NaN.")
                df[col] = pd.NA
            # Attempt to cast to specified type, allowing errors to become NaT/NaN
            try:
                if pd.api.types.is_datetime64_any_dtype(col_type): # Placeholder for datetime types if specified
                     df[col] = pd.to_datetime(df[col], errors='coerce')
                elif col_type == int: # More robust casting for integers that might be float
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64') # Using Int64 to support NA
                else:
                    df[col] = df[col].astype(col_type, errors='ignore') # errors='ignore' for flexibility with real data
            except Exception as e:
                print(f"Warning: Could not cast column '{col}' in {table_name}.csv to {col_type}. Error: {e}")
        
        print(f"Successfully loaded {table_name}.csv with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame(columns=schema.keys())

def load_all_data(data_dir):
    """Loads all specified Moodle log tables."""
    # --- Thesis Section 3.3.1: Data Log Moodle Riil ---
    # --- Thesis Section 3.3.2: Data Artifisial ---
    print(f"Loading all datasets from: {data_dir}")
    all_dfs = {}
    for table_name, schema in EXPECTED_TABLES.items():
        all_dfs[table_name] = load_csv_table(data_dir, table_name, schema)
    
    # Example of a basic referential integrity check (can be expanded)
    if not all_dfs["mdl_quiz_attempts"].empty and not all_dfs["mdl_user"].empty:
        missing_users = all_dfs["mdl_quiz_attempts"][~all_dfs["mdl_quiz_attempts"]["user_id"].isin(all_dfs["mdl_user"]["user_id"])]["user_id"].nunique()
        if missing_users > 0:
            print(f"Warning: {missing_users} unique user_ids in mdl_quiz_attempts are not in mdl_user.")
            
    return all_dfs

if __name__ == '__main__':
    # Example usage:
    # Replace with the actual path to your data
    # For generated data: 'data/moodle_logs_final_viz_corrected/'
    # For real data: path_to_real_fasilkom_logs
    data_directory = 'data/moodle_logs_final_viz_corrected/' 
    loaded_data = load_all_data(data_directory)
    
    # print("\nSummary of loaded data:")
    # for name, df in loaded_data.items():
    #     print(f"\nTable: {name}")
    #     df.info()
    #     if not df.empty:
    #         print(df.head())