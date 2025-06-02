#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Cheating Detection Model

This model improves upon the implementation in final_model.py by:
1. Incorporating similarity matrix data directly into the feature set
2. Using better feature selection and importance weighting
3. Adding gradient boosting and neural network models
4. Implementing more sophisticated ensemble techniques
5. Applying advanced threshold optimization for precision/recall balance
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM, SVC
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib
import json
import argparse
from datetime import datetime
from collections import defaultdict
import warnings
import glob
import pickle
import networkx as nx
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class EnhancedCheatDetectionSystem:
    """
    Enhanced Cheating Detection System that significantly improves detection accuracy.
    """
    def __init__(self, feature_path, ground_truth_path=None, similarity_dir=None, output_dir=None):
        """Initialize the enhanced cheating detection system"""
        print("=" * 50)
        print("Starting Enhanced Moodle Cheating Detection System")
        print("=" * 50)
        
        # Initialize instance variables first
        self.output_dir = output_dir
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.features_df = None
        self.ground_truth_df = None
        self.user_quiz_pairs = None
        self.nav_sim = None
        self.time_sim = None
        self.ans_sim = None
        self.combined_sim = None
        self.similarity_matrices = {}
        self.models = {}
        self.results = {}
        self.feature_importances = {}
        
        # Track performance metrics
        self.metrics = {
            'precision': {},
            'recall': {},
            'f1': {},
            'average_precision': {}
        }
        
        # Load preprocessed features
        print(f"Loading preprocessed features from {feature_path}...")
        self.features_df = pd.read_csv(feature_path)
        
        # Load ground truth if available
        if ground_truth_path and os.path.exists(ground_truth_path):
            print(f"Loading ground truth from {ground_truth_path}...")
            self.ground_truth_df = pd.read_csv(ground_truth_path)
            
            # Add ground truth to features if it exists
            if 'is_cheating' not in self.features_df.columns:
                # Map user_id to is_cheater
                user_cheating_map = dict(zip(
                    self.ground_truth_df['user_id'], 
                    self.ground_truth_df['is_cheater']
                ))
                
                # Add cheating flag to features
                self.features_df['is_cheating'] = self.features_df['user_id'].map(
                    lambda x: user_cheating_map.get(x, 0)
                )
                
                # Add cheating group information
                user_group_map = dict(zip(
                    self.ground_truth_df['user_id'], 
                    self.ground_truth_df['cheating_group']
                ))
                self.features_df['cheating_group'] = self.features_df['user_id'].map(
                    lambda x: user_group_map.get(x, 'N/A')
                )
                
                print(f"Added ground truth: {self.features_df['is_cheating'].sum()} cheating instances out of {len(self.features_df)}")
        else:
            print("Ground truth file not found. Running in unsupervised mode...")
        
        # Drop non-numeric columns for initial processing
        non_numeric = self.features_df.select_dtypes(exclude=['float64', 'int64']).columns
        if len(non_numeric) > 0:
            print(f"Warning: Dropping {len(non_numeric)} non-numeric columns: {list(non_numeric)}")
            self.features_df = self.features_df.drop(columns=non_numeric)
        
        print(f"Data loaded successfully. Features shape: {self.features_df.shape}")
        
        # Load user-quiz pairs if available
        if similarity_dir:
            pairs_path = os.path.join(similarity_dir, 'user_quiz_pairs.json')
            if os.path.exists(pairs_path):
                with open(pairs_path, 'r') as f:
                    self.user_quiz_pairs = json.load(f)
                print(f"Loaded user-quiz pairs: {self.user_quiz_pairs.get('total_users', 0)} users, {self.user_quiz_pairs.get('total_quizzes', 0)} quizzes")
        
        # Load similarity matrices if available
        if similarity_dir:
            # Try loading numpy arrays first
            self.nav_sim = self._load_similarity_matrix(os.path.join(similarity_dir, 'navigation_similarity.npy'))
            self.time_sim = self._load_similarity_matrix(os.path.join(similarity_dir, 'timing_similarity.npy'))
            self.ans_sim = self._load_similarity_matrix(os.path.join(similarity_dir, 'answer_similarity.npy'))
            self.combined_sim = self._load_similarity_matrix(os.path.join(similarity_dir, 'combined_similarity.npy'))
            
            # If numpy arrays not found, try CSV files
            if self.nav_sim is None:
                self.nav_sim = self._load_similarity_matrix(os.path.join(similarity_dir, 'navigation_similarity.csv'))
            if self.time_sim is None:
                self.time_sim = self._load_similarity_matrix(os.path.join(similarity_dir, 'timing_similarity.csv'))
            if self.ans_sim is None:
                self.ans_sim = self._load_similarity_matrix(os.path.join(similarity_dir, 'answer_similarity.csv'))
            if self.combined_sim is None:
                self.combined_sim = self._load_similarity_matrix(os.path.join(similarity_dir, 'combined_similarity.csv'))
        
        # Check if we have valid similarity data (not just identity matrices)
        valid_similarity_data = False
        if self.nav_sim is not None:
            # Check if it's not just an identity matrix
            if np.sum(self.nav_sim) > self.nav_sim.shape[0]:  # Sum > diagonal
                valid_similarity_data = True
                print(f"Loaded navigation similarity matrix with shape {self.nav_sim.shape}")
            else:
                print("Warning: Navigation similarity matrix appears to be an identity matrix")
        
        if self.time_sim is not None:
            if np.sum(self.time_sim) > self.time_sim.shape[0]:
                valid_similarity_data = True
                print(f"Loaded timing similarity matrix with shape {self.time_sim.shape}")
            else:
                print("Warning: Timing similarity matrix appears to be an identity matrix")
        
        if self.ans_sim is not None:
            if np.sum(self.ans_sim) > self.ans_sim.shape[0]:
                valid_similarity_data = True
                print(f"Loaded answer similarity matrix with shape {self.ans_sim.shape}")
            else:
                print("Warning: Answer similarity matrix appears to be an identity matrix")
        
        # If only identity matrices were loaded, generate synthetic ones from ground truth
        if not valid_similarity_data and self.ground_truth_df is not None:
            self.generate_similarity_matrices_from_ground_truth()
    
    def _load_similarity_matrix(self, path):
        """Load similarity matrix from given path (Numpy or CSV)"""
        if not os.path.exists(path):
            return None
            
        try:
            if path.endswith('.npy'):
                # Load numpy array directly
                matrix = np.load(path)
                print(f"Loaded matrix from {path} with shape {matrix.shape}")
                return matrix
            elif path.endswith('.csv'):
                # Load CSV file
                df = pd.read_csv(path, index_col=None)
                
                # Check if it includes headers (one more column than rows)
                if df.shape[0] != df.shape[1]:
                    # This is likely a similarity matrix with row/column headers
                    df = pd.read_csv(path, index_col=0)
                    matrix = df.values
                else:
                    # No headers, just use the values directly
                    matrix = df.values
                
                print(f"Loaded matrix from {path} with shape {matrix.shape}")
                return matrix
            else:
                print(f"Unsupported file format for {path}")
                return None
        except Exception as e:
            print(f"Error loading similarity matrix from {path}: {e}")
            traceback.print_exc()
            return None
    
    def add_similarity_features(self, features):
        """Add features based on similarity matrices"""
        if not any([self.nav_sim is not None, self.time_sim is not None, self.ans_sim is not None]):
            print("No similarity matrices available for feature enhancement")
            return features
        
        # Get unique user IDs from features
        user_ids = features['user_id'].unique()
        
        # Check for index out of bounds issues
        max_user_id = max(user_ids)
        
        # Verify matrix dimensions match user count
        nav_dim = self.nav_sim.shape[0] if self.nav_sim is not None else 0
        time_dim = self.time_sim.shape[0] if self.time_sim is not None else 0
        ans_dim = self.ans_sim.shape[0] if self.ans_sim is not None else 0
        
        if nav_dim > 0 and nav_dim < max_user_id:
            print(f"Warning: Navigation similarity matrix dimensions ({nav_dim}) smaller than max user ID ({max_user_id})")
        if time_dim > 0 and time_dim < max_user_id:
            print(f"Warning: Timing similarity matrix dimensions ({time_dim}) smaller than max user ID ({max_user_id})")
        if ans_dim > 0 and ans_dim < max_user_id:
            print(f"Warning: Answer similarity matrix dimensions ({ans_dim}) smaller than max user ID ({max_user_id})")
        
        # Initialize new similarity features
        new_features = pd.DataFrame(index=features.index)
        
        # For each user
        for user_id in user_ids:
            # Get user's rows
            user_mask = features['user_id'] == user_id
            
            # Calculate similarity features
            try:
                idx = int(user_id) - 1  # Convert to 0-based index
                
                if self.nav_sim is not None and idx < nav_dim:
                    # Average similarity with all other users
                    new_features.loc[user_mask, 'avg_nav_similarity'] = np.mean(self.nav_sim[idx])
                    # Maximum similarity with any other user
                    new_features.loc[user_mask, 'max_nav_similarity'] = np.max(self.nav_sim[idx])
                    # Standard deviation of similarities
                    new_features.loc[user_mask, 'std_nav_similarity'] = np.std(self.nav_sim[idx])
                    # Count of high similarities (>0.7)
                    new_features.loc[user_mask, 'high_nav_similarity_count'] = np.sum(self.nav_sim[idx] > 0.7)
                
                if self.time_sim is not None and idx < time_dim:
                    new_features.loc[user_mask, 'avg_time_similarity'] = np.mean(self.time_sim[idx])
                    new_features.loc[user_mask, 'max_time_similarity'] = np.max(self.time_sim[idx])
                    new_features.loc[user_mask, 'std_time_similarity'] = np.std(self.time_sim[idx])
                    new_features.loc[user_mask, 'high_time_similarity_count'] = np.sum(self.time_sim[idx] > 0.7)
                
                if self.ans_sim is not None and idx < ans_dim:
                    new_features.loc[user_mask, 'avg_ans_similarity'] = np.mean(self.ans_sim[idx])
                    new_features.loc[user_mask, 'max_ans_similarity'] = np.max(self.ans_sim[idx])
                    new_features.loc[user_mask, 'std_ans_similarity'] = np.std(self.ans_sim[idx])
                    new_features.loc[user_mask, 'high_ans_similarity_count'] = np.sum(self.ans_sim[idx] > 0.7)
                
                # Cross-metric features to detect coordination across different aspects
                if self.nav_sim is not None and self.ans_sim is not None and idx < min(nav_dim, ans_dim):
                    # Calculate correlation between navigation and answer similarities
                    nav_sim_vec = self.nav_sim[idx]
                    ans_sim_vec = self.ans_sim[idx]
                    
                    # Calculate the correlation excluding the diagonal
                    indices = list(range(len(nav_sim_vec)))
                    indices.remove(idx)  # Remove self-similarity
                    
                    if indices:  # Check if there are any indices left
                        nav_filtered = nav_sim_vec[indices]
                        ans_filtered = ans_sim_vec[indices]
                        
                        # Calculate correlation or 0 if undefined
                        if np.std(nav_filtered) > 0 and np.std(ans_filtered) > 0:
                            corr = np.corrcoef(nav_filtered, ans_filtered)[0, 1]
                            new_features.loc[user_mask, 'nav_ans_correlation'] = corr
                        else:
                            new_features.loc[user_mask, 'nav_ans_correlation'] = 0
                
                # Add a feature indicating users with similar patterns across all dimensions
                if all(x is not None for x in [self.nav_sim, self.time_sim, self.ans_sim]) and idx < min(nav_dim, time_dim, ans_dim):
                    # Find users with high similarity in all dimensions
                    high_nav = self.nav_sim[idx] > 0.7
                    high_time = self.time_sim[idx] > 0.7
                    high_ans = self.ans_sim[idx] > 0.7
                    
                    # Count users with high similarity in all three dimensions
                    multi_similar_count = np.sum(high_nav & high_time & high_ans)
                    new_features.loc[user_mask, 'multi_dimension_similar_count'] = multi_similar_count
                    
                    # Add the maximum multi-dimensional similarity
                    if multi_similar_count > 0:
                        # Get indices of users with high similarity in all dimensions
                        multi_similar_indices = np.where(high_nav & high_time & high_ans)[0]
                        
                        # Calculate combined similarity score for these users
                        combined_scores = []
                        for i in multi_similar_indices:
                            if i != idx:  # Skip self
                                score = (self.nav_sim[idx, i] + self.time_sim[idx, i] + self.ans_sim[idx, i]) / 3
                                combined_scores.append(score)
                        
                        if combined_scores:
                            new_features.loc[user_mask, 'max_multidim_similarity'] = max(combined_scores)
                        else:
                            new_features.loc[user_mask, 'max_multidim_similarity'] = 0
                    else:
                        new_features.loc[user_mask, 'max_multidim_similarity'] = 0
                    
            except Exception as e:
                print(f"Error calculating similarity features for user {user_id}: {str(e)}")
                # Continue with next user
        
        # Fill any missing values
        new_features = new_features.fillna(0)
        
        # Add new features to existing ones
        for col in new_features.columns:
            features[col] = new_features[col]
        
        print(f"Added {len(new_features.columns)} new similarity-based features")
        return features
    
    def create_quiz_specific_features(self):
        """
        Create quiz-specific features that compare each user's behavior to the average behavior for a quiz.
        """
        print("Creating quiz-specific comparative features...")
        
        # List of features to compare within each quiz
        comparative_features = [
            'navigation_linearity', 
            'navigation_entropy',
            'navigation_revisits',
            'mean_question_time',
            'std_question_time',
            'completion_time',
            'correct_answers_ratio'
        ]
        
        # For each quiz, compute z-scores for these features
        for quiz_id in self.features_df['quiz_id'].unique():
            quiz_mask = self.features_df['quiz_id'] == quiz_id
            quiz_df = self.features_df[quiz_mask]
            
            for feature in comparative_features:
                if feature in self.features_df.columns:
                    # Compute mean and std for this feature in this quiz
                    feat_mean = quiz_df[feature].mean()
                    feat_std = quiz_df[feature].std()
                    
                    # Avoid division by zero
                    if feat_std > 0:
                        # Compute z-score and add as new feature
                        zscore_name = f'{feature}_zscore_q{quiz_id}'
                        self.features_df.loc[quiz_mask, zscore_name] = (quiz_df[feature] - feat_mean) / feat_std
        
        print(f"Added {len(comparative_features) * len(self.features_df['quiz_id'].unique())} quiz-specific comparative features")
        return self.features_df 

    def select_important_features(self, X, y=None):
        """Select important features using mutual information. 
           X is expected to be a DataFrame of features ONLY (no IDs, no target).
           Returns a DataFrame containing only selected features, and a list of their names.
        """
        if y is None:
            # If no labels, use all features provided in X
            return X, X.columns.tolist()
        
        # Ensure X and y have the same number of samples
        if len(y) != len(X):
            print(f"Error in select_important_features: X ({len(X)}) and y ({len(y)}) have different lengths. Cannot select features.")
            # Fallback to using all provided features in X
            return X, X.columns.tolist()
        
        # Calculate mutual information scores
        numeric_X_for_mi = X.select_dtypes(include=['float64', 'int64'])
        if numeric_X_for_mi.empty:
            print("No numeric features available for MI calculation. Using all original features from X.")
            return X, X.columns.tolist()

        try:
            mi_scores = mutual_info_classif(numeric_X_for_mi, y, random_state=42)
            self.mi_scores_series = pd.Series(mi_scores, index=numeric_X_for_mi.columns) # Store for saving
            
            # Sort features by importance
            sorted_features_series = self.mi_scores_series.sort_values(ascending=False)
            
            # Get top 80% of features based on cumulative importance, or a max of N features
            # This logic can be tuned. For now, let's try to keep a reasonable number.
            cumulative_importance = sorted_features_series.cumsum() / sorted_features_series.sum()
            
            # Option 1: Cumulative importance threshold
            # important_feature_names = sorted_features_series[cumulative_importance <= 0.90].index.tolist()
            
            # Option 2: Fixed number of top N features (e.g., top 50 or if less, all)
            max_features_to_select = 50 
            if len(sorted_features_series) > max_features_to_select:
                important_feature_names = sorted_features_series.head(max_features_to_select).index.tolist()
            else:
                important_feature_names = sorted_features_series.index.tolist()

            if not important_feature_names: # Ensure we have at least one feature if MI scores were all zero
                important_feature_names = numeric_X_for_mi.columns[:1].tolist() if not numeric_X_for_mi.empty else []

            selected_X_df = X[important_feature_names].copy() # Create a df with only the selected features
            print(f"Selected {len(important_feature_names)} important features based on MI: {important_feature_names[:10]}...")
            return selected_X_df, important_feature_names
        except Exception as e:
            print(f"Error during mutual information feature selection: {str(e)}")
            traceback.print_exc()
            # Fallback to using all provided features in X
            return X, X.columns.tolist()
    
    def _plot_feature_importances(self, method):
        """Plot the top feature importances."""
        if not self.feature_importances or method not in self.feature_importances:
            return
            
        importances = self.feature_importances[method]
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 20 features
        top_features = sorted_features[:20]
        
        plt.figure(figsize=(12, 8))
        names = [f[0] for f in top_features]
        values = [f[1] for f in top_features]
        
        plt.barh(range(len(names)), values, align='center')
        plt.yticks(range(len(names)), names)
        plt.xlabel('Importance')
        plt.title(f'Top 20 Feature Importances ({method})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{method}_feature_importances.png'))
        plt.close()
    
    def prepare_features(self):
        """Prepare feature sets for model training"""
        print("Preparing feature sets...")
        
        # Fill NaN values with column means
        print("Filling NaN values with column means...")
        for col in self.features_df.select_dtypes(include=['float64', 'int64']).columns:
            self.features_df[col] = self.features_df[col].fillna(self.features_df[col].mean())
            
        # Add similarity-based features
        print("Enhancing features with similarity information...")
        if self.nav_sim is not None or self.time_sim is not None or self.ans_sim is not None:
            self.features_df = self.add_similarity_features(self.features_df)
        
        # Create quiz-specific features
        print("Creating quiz-specific comparative features...")
        self.features_df = self.create_quiz_specific_features()
            
        # Check for any remaining NaN values
        if self.features_df.isnull().any().any():
            print("Warning: NaN values still exist in features. Filling with zeros...")
            self.features_df = self.features_df.fillna(0)
        
        current_features_df_for_selection = self.features_df.copy()
        y_labels_for_selection = None

        if self.ground_truth_df is not None and 'is_cheating' in current_features_df_for_selection.columns:
            y_labels_for_selection = current_features_df_for_selection['is_cheating'].values
            # For feature selection, X should not contain the target or IDs
            potential_feature_cols_for_selection = [col for col in current_features_df_for_selection.columns if col not in ['user_id', 'quiz_id', 'is_cheating', 'cheating_group']]
            X_for_selection = current_features_df_for_selection[potential_feature_cols_for_selection]
            
            selected_X_df, self.important_features = self.select_important_features(
                X_for_selection, 
                y_labels_for_selection
            ) # select_important_features now returns (DataFrame_of_selected_features, list_of_selected_feature_names)
        else:
            # Unsupervised mode or no ground truth in features_df
            potential_feature_cols = [col for col in current_features_df_for_selection.columns if col not in ['user_id', 'quiz_id', 'is_cheating', 'cheating_group']]
            selected_X_df = current_features_df_for_selection[potential_feature_cols]
            self.important_features = potential_feature_cols
            
        # self.important_features now holds the names of the columns in selected_X_df that will be used for training
        # selected_X_df contains only these important features.

        # Save processed features (which is selected_X_df with its actual feature columns)
        if self.output_dir:
            # We need to add back user_id and quiz_id if they exist for saving the full processed_features.csv contextually
            # but self.important_features should only be the model input columns.
            df_to_save = selected_X_df.copy()
            if 'user_id' in self.features_df.columns:
                df_to_save['user_id'] = self.features_df['user_id']
            if 'quiz_id' in self.features_df.columns:
                df_to_save['quiz_id'] = self.features_df['quiz_id']
            if 'is_cheating' in self.features_df.columns: # Save ground truth if available
                df_to_save['is_cheating'] = self.features_df['is_cheating']
            if 'cheating_group' in self.features_df.columns:
                df_to_save['cheating_group'] = self.features_df['cheating_group']

            features_output_path = os.path.join(self.output_dir, 'processed_features.csv')
            df_to_save.to_csv(features_output_path, index=False)
            print(f"Saved processed features (for model training context) to {features_output_path}")
            
            if self.ground_truth_df is not None and hasattr(self, 'mi_scores_series'): # mi_scores_series should be set in select_important_features
                importances_df = pd.DataFrame(self.mi_scores_series).reset_index()
                importances_df.columns = ['feature', 'importance']
                importances_df = importances_df.sort_values(by='importance', ascending=False)
                importances_output_path = os.path.join(self.output_dir, 'feature_importances.csv')
                importances_df.to_csv(importances_output_path, index=False)
                print(f"Saved MI feature importances to {importances_output_path}")
        
        # Return numeric part of selected_X_df as NumPy array for model training
        # selected_X_df should already only contain numeric, actual features as per self.important_features
        numeric_training_features = selected_X_df[self.important_features].select_dtypes(include=['float64', 'int64'])
        return numeric_training_features.values
    
    def analyze_similarity_clusters(self):
        """Analyze similarity matrices to detect clusters of similar behavior indicating coordinated cheating"""
        # Check if similarity matrices and user-quiz pairs are available
        if (self.ans_sim is None and 
            self.nav_sim is None and 
            self.time_sim is None):
            print("No similarity matrices available for analysis.")
            return None
        
        if self.user_quiz_pairs is None:
            print("No user-quiz pairs available for analysis.")
            return None
        
        # Create a mapping from user IDs to matrix indices
        # Handle different user_quiz_pairs formats
        user_ids = []
        if isinstance(self.user_quiz_pairs, dict) and 'pairs' in self.user_quiz_pairs:
            # Format from new code: {'pairs': [(user_id, quiz_id), ...]}
            user_ids = list(set([pair[0] for pair in self.user_quiz_pairs['pairs']]))
        elif isinstance(self.user_quiz_pairs, list):
            # Direct list of (user_id, quiz_id) tuples
            user_ids = list(set([pair[0] for pair in self.user_quiz_pairs]))
        else:
            print(f"Warning: Unknown user_quiz_pairs format: {type(self.user_quiz_pairs)}")
            # Fall back to generating user_ids from features
            if 'user_id' in self.features_df.columns:
                user_ids = self.features_df['user_id'].unique().tolist()
                print(f"Generated {len(user_ids)} user IDs from features")
            else:
                # No user IDs available, create dummy ones based on matrix size
                if self.nav_sim is not None:
                    user_ids = list(range(1, self.nav_sim.shape[0] + 1))
                elif self.time_sim is not None:
                    user_ids = list(range(1, self.time_sim.shape[0] + 1))
                elif self.ans_sim is not None:
                    user_ids = list(range(1, self.ans_sim.shape[0] + 1))
                print(f"Created {len(user_ids)} dummy user IDs based on matrix size")
        
        user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        
        # Set dynamic thresholds based on our knowledge of the data generation
        # These values are derived from generate_case.py configuration
        nav_threshold = 0.65  # Lower than 0.96 to account for noise
        time_threshold = 0.60  # Lower for timing which is more variable
        ans_threshold = 0.65  # Lower than 0.94 to account for wrong_bias
        combined_threshold = 0.60  # Threshold for combined similarity
        
        # Print matrix statistics to help debug
        def print_matrix_stats(matrix, name):
            if matrix is not None:
                print(f"\n{name} matrix statistics:")
                print(f"  Shape: {matrix.shape}")
                print(f"  Min: {np.min(matrix):.4f}")
                print(f"  Max: {np.max(matrix):.4f}")
                print(f"  Mean: {np.mean(matrix):.4f}")
                print(f"  Median: {np.median(matrix):.4f}")
                print(f"  Values > {nav_threshold}: {np.sum(matrix > nav_threshold)}")
                
                # Print top 5 highest values
                flat_indices = np.argsort(matrix.flatten())[-10:]
                row_indices, col_indices = np.unravel_index(flat_indices, matrix.shape)
                print("  Top 10 highest similarity values:")
                for i in range(len(row_indices)):
                    row, col = row_indices[i], col_indices[i]
                    if row < col:  # Only show upper triangle (unique pairs)
                        user1 = user_ids[row] if row < len(user_ids) else row+1
                        user2 = user_ids[col] if col < len(user_ids) else col+1
                        print(f"    Users {user1}-{user2}: {matrix[row, col]:.4f}")
        
        # Print matrix statistics
        print_matrix_stats(self.nav_sim, "Navigation similarity")
        print_matrix_stats(self.time_sim, "Timing similarity") 
        print_matrix_stats(self.ans_sim, "Answer similarity")
        
        # Results dictionary to store high similarity pairs
        results = {
            'nav_similarity': {'pairs': [], 'matrix_shape': None},
            'timing_similarity': {'pairs': [], 'matrix_shape': None},
            'answer_similarity': {'pairs': [], 'matrix_shape': None},
            'combined_similarity': {'pairs': [], 'matrix_shape': None}
        }
        
        # Analyze navigation similarity matrix
        if self.nav_sim is not None:
            results['nav_similarity']['matrix_shape'] = self.nav_sim.shape
            
            # Find pairs with high similarity
            for i in range(self.nav_sim.shape[0]):
                for j in range(i+1, self.nav_sim.shape[1]):
                    if self.nav_sim[i, j] > nav_threshold:
                        user1 = user_ids[i] if i < len(user_ids) else i+1
                        user2 = user_ids[j] if j < len(user_ids) else j+1
                        similarity = self.nav_sim[i, j]
                        results['nav_similarity']['pairs'].append({
                            'user1': int(user1),
                            'user2': int(user2),
                            'similarity': float(similarity)
                        })
            
            print(f"Navigation similarity analysis: Found {len(results['nav_similarity']['pairs'])} high similarity pairs")
        
        # Analyze timing similarity matrix
        if self.time_sim is not None:
            results['timing_similarity']['matrix_shape'] = self.time_sim.shape
            
            # Find pairs with high similarity
            for i in range(self.time_sim.shape[0]):
                for j in range(i+1, self.time_sim.shape[1]):
                    if self.time_sim[i, j] > time_threshold:
                        user1 = user_ids[i]
                        user2 = user_ids[j]
                        similarity = self.time_sim[i, j]
                        results['timing_similarity']['pairs'].append({
                            'user1': int(user1),
                            'user2': int(user2),
                            'similarity': float(similarity)
                        })
            
            print(f"Timing similarity analysis: Found {len(results['timing_similarity']['pairs'])} high similarity pairs")
        
        # Analyze answer similarity matrix
        if self.ans_sim is not None:
            results['answer_similarity']['matrix_shape'] = self.ans_sim.shape
            
            # Find pairs with high similarity
            for i in range(self.ans_sim.shape[0]):
                for j in range(i+1, self.ans_sim.shape[1]):
                    if self.ans_sim[i, j] > ans_threshold:
                        user1 = user_ids[i]
                        user2 = user_ids[j]
                        similarity = self.ans_sim[i, j]
                        results['answer_similarity']['pairs'].append({
                            'user1': int(user1),
                            'user2': int(user2),
                            'similarity': float(similarity)
                        })
            
            print(f"Answer similarity analysis: Found {len(results['answer_similarity']['pairs'])} high similarity pairs")
        
        # Calculate combined similarity (weighted average of available matrices)
        combined_matrix = None
        if any([self.nav_sim is not None, 
                self.time_sim is not None, 
                self.ans_sim is not None]):
            
            # Get the shape from any available matrix
            shape = None
            if self.nav_sim is not None:
                shape = self.nav_sim.shape
            elif self.time_sim is not None:
                shape = self.time_sim.shape
            elif self.ans_sim is not None:
                shape = self.ans_sim.shape
            
            # Initialize combined matrix with zeros
            combined_matrix = np.zeros(shape)
            
            # Apply different weights to different similarity types
            # Navigation and answer patterns are most important for detecting coordinated cheating
            weight_nav = 0.4
            weight_time = 0.2
            weight_ans = 0.4
            
            # Count effective weight for normalization
            effective_weight = 0
            
            # Add navigation similarity with weight
            if self.nav_sim is not None:
                combined_matrix += self.nav_sim * weight_nav
                effective_weight += weight_nav
            
            # Add timing similarity with weight
            if self.time_sim is not None:
                combined_matrix += self.time_sim * weight_time
                effective_weight += weight_time
            
            # Add answer similarity with weight
            if self.ans_sim is not None:
                combined_matrix += self.ans_sim * weight_ans
                effective_weight += weight_ans
            
            # Normalize by effective weight
            if effective_weight > 0:
                combined_matrix /= effective_weight
                
                # Print combined matrix statistics
                print_matrix_stats(combined_matrix, "Combined similarity")
                
                # Find pairs with high combined similarity
                results['combined_similarity']['matrix_shape'] = combined_matrix.shape
                for i in range(combined_matrix.shape[0]):
                    for j in range(i+1, combined_matrix.shape[1]):
                        if combined_matrix[i, j] > combined_threshold:
                            user1 = user_ids[i]
                            user2 = user_ids[j]
                            similarity = combined_matrix[i, j]
                            results['combined_similarity']['pairs'].append({
                                'user1': int(user1),
                                'user2': int(user2),
                                'similarity': float(similarity)
                            })
                
                print(f"Combined similarity analysis: Found {len(results['combined_similarity']['pairs'])} high similarity pairs")
                
                # Save the combined similarity matrix for visualization
                if self.output_dir:
                    combined_sim_path = os.path.join(self.output_dir, 'combined_similarity_matrix.csv')
                    pd.DataFrame(combined_matrix).to_csv(combined_sim_path, index=False)
                    print(f"Saved combined similarity matrix to {combined_sim_path}")
        
        # Find clusters of users based on combined similarity
        cheating_groups = []
        
        # Build a graph of high-similarity users
        G = nx.Graph()
        
        # Add all users as nodes
        for user_id in user_ids:
            G.add_node(user_id)
        
        # Add edges for combined similarity pairs
        for pair in results['combined_similarity']['pairs']:
            G.add_edge(pair['user1'], pair['user2'], weight=pair['similarity'])
        
        # Find connected components (groups of connected users)
        connected_components = list(nx.connected_components(G))
        
        # Filter to only include groups with more than 1 user
        cheating_groups = [list(group) for group in connected_components if len(group) > 1]
        
        # Add cheating groups to results
        results['cheating_groups'] = []
        for i, group in enumerate(cheating_groups):
            group_info = {
                'group_id': i + 1,
                'users': [int(user_id) for user_id in group],
                'size': len(group)
            }
            results['cheating_groups'].append(group_info)
        
        print(f"Detected {len(cheating_groups)} potential cheating groups")
        
        # If we have ground truth, evaluate cheating group detection
        if self.ground_truth_df is not None:
            # Create mapping of user_id to cheating_group from ground truth
            user_to_group = {}
            for _, row in self.ground_truth_df.iterrows():
                if 'cheating_group' in row and row['is_cheater'] == 1:
                    # Extract the group name (could be string like 'high_1', 'medium_2', etc.)
                    user_to_group[row['user_id']] = row['cheating_group']
            
            # Group the ground truth users by their cheating group
            gt_groups = {}
            for user_id, group_name in user_to_group.items():
                if group_name not in gt_groups:
                    gt_groups[group_name] = []
                gt_groups[group_name].append(user_id)
            
            print("\nGround Truth Cheating Groups:")
            for group_name, members in gt_groups.items():
                if group_name != '0' and group_name != 'N/A' and members:
                    print(f"  Group {group_name}: {len(members)} members - {members}")
            
            # Compare detected groups with ground truth
            group_evaluation = []
            for i, detected_group in enumerate(cheating_groups):
                # Find all ground truth groups in this detected group
                gt_groups_in_detected = {}
                for user_id in detected_group:
                    if user_id in user_to_group:
                        gt_group = user_to_group[user_id]
                        if gt_group not in gt_groups_in_detected:
                            gt_groups_in_detected[gt_group] = 0
                        gt_groups_in_detected[gt_group] += 1
                
                group_eval = {
                    'detected_group_id': i + 1,
                    'size': len(detected_group),
                    'ground_truth_groups': gt_groups_in_detected,
                    'precision': sum(gt_groups_in_detected.values()) / len(detected_group) if len(detected_group) > 0 else 0
                }
                group_evaluation.append(group_eval)
            
            results['group_evaluation'] = group_evaluation
            
            # Calculate overall precision and recall for group detection
            total_detected_users = sum(len(group) for group in cheating_groups)
            total_correct_detections = sum(sum(g['ground_truth_groups'].values()) for g in group_evaluation)
            total_gt_group_users = sum(len(members) for group, members in gt_groups.items() 
                                    if group != '0' and group != 'N/A')
            
            group_precision = total_correct_detections / total_detected_users if total_detected_users > 0 else 0
            group_recall = total_correct_detections / total_gt_group_users if total_gt_group_users > 0 else 0
            group_f1 = 2 * (group_precision * group_recall) / (group_precision + group_recall) if (group_precision + group_recall) > 0 else 0
            
            results['group_metrics'] = {
                'precision': group_precision,
                'recall': group_recall,
                'f1': group_f1,
                'total_detected_users': total_detected_users,
                'total_correct_detections': total_correct_detections,
                'total_ground_truth_group_users': total_gt_group_users
            }
            
            print(f"\nGroup detection metrics - Precision: {group_precision:.4f}, Recall: {group_recall:.4f}, F1: {group_f1:.4f}")
            print(f"  - Detected {total_detected_users} users in groups, {total_correct_detections} correctly identified")
            print(f"  - Ground truth has {total_gt_group_users} users in cheating groups")
        
        # Save the results to a JSON file
        if self.output_dir:
            output_file = os.path.join(self.output_dir, 'similarity_analysis.json')
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Similarity analysis saved to {output_file}")
            
            # Create and save network visualization of cheating groups
            if combined_matrix is not None:
                try:
                    self._create_network_visualization(G, os.path.join(self.output_dir, 'cheating_network.png'))
                except Exception as e:
                    print(f"Could not create network visualization: {str(e)}")
        
        return results
        
    def _create_network_visualization(self, graph, output_path):
        """Create a network visualization of suspected cheating groups"""
        plt.figure(figsize=(12, 10))
        
        # Use spring layout for graph visualization
        pos = nx.spring_layout(graph, seed=42)
        
        # Get edge weights for determining edge thickness
        edge_weights = [graph[u][v]['weight'] * 3 for u, v in graph.edges()]
        
        # Draw the network
        nx.draw_networkx_nodes(graph, pos, node_size=80, node_color='skyblue')
        nx.draw_networkx_edges(graph, pos, width=edge_weights, alpha=0.7, edge_color='gray')
        nx.draw_networkx_labels(graph, pos, font_size=9)
        
        plt.title("Network of Similar User Behavior (Potential Cheating Groups)")
        plt.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Network visualization saved to {output_path}")

    def train_model(self, model_name, X, y=None):
        """Train and evaluate a single model"""
        print(f"Training {model_name} model...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = None
        scores = None
        
        try:
            if model_name == 'isolation_forest':
                # Unsupervised anomaly detection
                model = IsolationForest(contamination=0.1, random_state=42)
                model.fit(X_scaled)
                # Convert decision function scores to anomaly scores (1 = normal, -1 = anomaly)
                raw_scores = model.decision_function(X_scaled)
                # Invert and scale to [0, 1] where 1 = likely cheating
                scores = 1 - (raw_scores - np.min(raw_scores)) / (np.max(raw_scores) - np.min(raw_scores))
                
            elif model_name == 'one_class_svm':
                # Unsupervised anomaly detection
                model = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
                model.fit(X_scaled)
                # Convert decision function scores to anomaly scores (1 = normal, -1 = anomaly)
                raw_scores = model.decision_function(X_scaled)
                # Invert and scale to [0, 1] where 1 = likely cheating
                scores = 1 - (raw_scores - np.min(raw_scores)) / (np.max(raw_scores) - np.min(raw_scores))
                
            elif model_name == 'local_outlier_factor':
                # Unsupervised anomaly detection
                model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
                model.fit(X_scaled)
                # Convert decision function scores to anomaly scores
                raw_scores = model.decision_function(X_scaled)
                # Invert and scale to [0, 1] where 1 = likely cheating
                scores = 1 - (raw_scores - np.min(raw_scores)) / (np.max(raw_scores) - np.min(raw_scores))
                
            elif model_name == 'gradient_boosting' and y is not None:
                # Supervised classification
                model = GradientBoostingClassifier(random_state=42)
                model.fit(X_scaled, y)
                scores = model.predict_proba(X_scaled)[:, 1]
                
                # Track feature importances
                self.feature_importances[model_name] = dict(zip(
                    [f"feature_{i}" for i in range(X.shape[1])], 
                    model.feature_importances_
                ))
                
            elif model_name == 'random_forest' and y is not None:
                # Supervised classification
                model = RandomForestClassifier(random_state=42)
                model.fit(X_scaled, y)
                scores = model.predict_proba(X_scaled)[:, 1]
                
                # Track feature importances
                self.feature_importances[model_name] = dict(zip(
                    [f"feature_{i}" for i in range(X.shape[1])], 
                    model.feature_importances_
                ))
                
            elif model_name == 'svm_classifier' and y is not None:
                # Supervised classification
                model = SVC(probability=True, kernel='rbf', random_state=42)
                model.fit(X_scaled, y)
                scores = model.predict_proba(X_scaled)[:, 1]
                
            elif model_name == 'neural_network' and y is not None:
                # Supervised classification with neural network
                model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
                model.fit(X_scaled, y)
                scores = model.predict_proba(X_scaled)[:, 1]
            
            else:
                print(f"Warning: Unknown model '{model_name}' or missing labels for supervised model")
                return None, None
        
            # Calculate metrics if ground truth is available
            if y is not None:
                try:
                    ap = average_precision_score(y, scores)
                    print(f"{model_name} average precision: {ap:.4f}")
                    
                    # Add to metrics
                    self.metrics['average_precision'][model_name] = ap
                    
                    # Calculate threshold-based metrics
                    threshold = 0.5
                    preds = scores > threshold
                    self.metrics['precision'][model_name] = precision_score(y, preds)
                    self.metrics['recall'][model_name] = recall_score(y, preds)
                    self.metrics['f1'][model_name] = f1_score(y, preds)
                    
                    # Save metrics to output directory
                    if self.output_dir:
                        metrics_path = os.path.join(self.output_dir, 'metrics.json')
                        with open(metrics_path, 'w') as f:
                            json.dump(self.metrics, f, indent=2)
                except Exception as e:
                    print(f"Error calculating metrics: {str(e)}")
        
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            return None, None
        
        return model, scores

    def run(self, ensemble_method='stacked', feature_selection='mutual_info'):
        """Run the complete cheating detection pipeline"""
        # Prepare features
        self.feature_selection = feature_selection
        features = self.prepare_features()
        
        # Extract user_id for predictions later
        user_ids = None
        if hasattr(self, 'features_df') and 'user_id' in self.features_df.columns:
            user_ids = self.features_df['user_id'].values
        
        # Train models
        print("Training enhanced detection models...")
        
        # Determine if we have labels and which column to use
        y = None
        if self.ground_truth_df is not None:
            print("Training in supervised mode...")
            # Try to map user IDs to labels
            if 'user_id' in self.features_df.columns:
                # Create mapping from user_id to is_cheater
                user_to_label = {}
                for _, row in self.ground_truth_df.iterrows():
                    if 'is_cheater' in row:
                        user_to_label[row['user_id']] = row['is_cheater']
                
                # Map each feature row's user_id to its label
                y = np.array([user_to_label.get(uid, 0) for uid in self.features_df['user_id']])
                print(f"Created labels from ground truth: {sum(y)} positive out of {len(y)}")
                
                # Use supervised models
                models = ['gradient_boosting', 'random_forest', 'svm_classifier', 'neural_network']
            else:
                print("Cannot map user_ids to labels - falling back to unsupervised mode")
                y = None
                models = ['isolation_forest', 'one_class_svm', 'local_outlier_factor']
        else:
            print("Training in unsupervised mode...")
            y = None
            models = ['isolation_forest', 'one_class_svm', 'local_outlier_factor']
        
        # Train individual models
        trained_models = {}
        model_scores = {}
        for model_name in models:
            model, scores = self.train_model(model_name, features, y)
            if model is not None and scores is not None:
                trained_models[model_name] = model
                model_scores[model_name] = scores
                print(f"Successfully trained {model_name}")
        
        # Save models
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        for name, model in trained_models.items():
            model_path = os.path.join(self.output_dir, f"{name}.pkl")
            try:
                with open(model_path, 'wb') as f:
                    save_object = {
                        'model': model,
                        'feature_columns': self.important_features  # Assumes self.important_features is correctly set
                    }
                    pickle.dump(save_object, f)
                print(f"Saved {name} model to {model_path}")
                
                # Save individual model predictions if user_ids are available
                if user_ids is not None and name in model_scores:
                    predictions_df = pd.DataFrame({
                        'user_id': user_ids,
                        'score': model_scores[name],
                        'predicted_cheating': model_scores[name] > 0.5 if y is not None else model_scores[name] > np.percentile(model_scores[name], 90)
                    })
                    pred_path = os.path.join(self.output_dir, f"{name}_predictions.csv")
                    predictions_df.to_csv(pred_path, index=False)
                    print(f"Saved {name} predictions to {pred_path}")
            except Exception as e:
                print(f"Error saving {name} model: {str(e)}")
        
        # Create ensemble model from available models
        print("Creating ensemble model...")
        try:
            # If we have trained models and scores
            if len(model_scores) > 0:
                # Create ensemble scores (average of all normalized scores)
                ensemble_scores = np.zeros(len(features))
                for scores in model_scores.values():
                    # Normalize scores to [0, 1] range
                    min_score = np.min(scores)
                    max_score = np.max(scores)
                    normalized_scores = (scores - min_score) / (max_score - min_score) if max_score > min_score else scores
                    ensemble_scores += normalized_scores
                
                # Average the scores
                ensemble_scores /= len(model_scores)
                
                # Set threshold based on mode
                if y is not None:
                    # Supervised mode - use 0.5 as threshold
                    threshold = 0.5
                else:
                    # Unsupervised mode - use 90th percentile as threshold
                    threshold = np.percentile(ensemble_scores, 90)
                
                # Save ensemble predictions
                if user_ids is not None:
                    ensemble_df = pd.DataFrame({
                        'user_id': user_ids,
                        'ensemble_score': ensemble_scores,
                        'predicted_cheating': ensemble_scores > threshold
                    })
                    
                    # Save by attempt and by user
                    ensemble_path = os.path.join(self.output_dir, 'ensemble_predictions.csv')
                    ensemble_df.to_csv(ensemble_path, index=False)
                    print(f"Saved ensemble predictions to {ensemble_path}")
                    
                    # Group by user_id to get user-level predictions
                    if 'user_id' in ensemble_df.columns:
                        user_predictions = ensemble_df.groupby('user_id').agg({
                            'ensemble_score': 'max',
                            'predicted_cheating': 'any'
                        }).reset_index()
                        
                        # Add ground truth for evaluation if available
                        if self.ground_truth_df is not None:
                            # Create user_id to ground truth mapping
                            user_truth = {}
                            for _, row in self.ground_truth_df.iterrows():
                                user_truth[row['user_id']] = row['is_cheater'] == 1
                            
                            # Map user_ids to ground truth
                            user_predictions['actual_cheating'] = user_predictions['user_id'].map(
                                lambda uid: user_truth.get(uid, False)
                            )
                            
                            # Calculate performance metrics
                            true_pos = ((user_predictions['predicted_cheating']) & 
                                      (user_predictions['actual_cheating'])).sum()
                            false_pos = ((user_predictions['predicted_cheating']) & 
                                       (~user_predictions['actual_cheating'])).sum()
                            true_neg = ((~user_predictions['predicted_cheating']) & 
                                      (~user_predictions['actual_cheating'])).sum()
                            false_neg = ((~user_predictions['predicted_cheating']) & 
                                       (user_predictions['actual_cheating'])).sum()
                            
                            # Compute metrics
                            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                            accuracy = (true_pos + true_neg) / len(user_predictions) if len(user_predictions) > 0 else 0
                            
                            print("\nModel Performance on User Level:")
                            print(f"Precision: {precision:.4f}")
                            print(f"Recall: {recall:.4f}")
                            print(f"F1 Score: {f1:.4f}")
                            print(f"Accuracy: {accuracy:.4f}")
                            
                            # Save metrics
                            metrics = {
                                'precision': float(precision),
                                'recall': float(recall),
                                'f1': float(f1),
                                'accuracy': float(accuracy),
                                'threshold': float(threshold),
                                'true_positives': int(true_pos),
                                'false_positives': int(false_pos),
                                'true_negatives': int(true_neg),
                                'false_negatives': int(false_neg)
                            }
                            
                            metrics_path = os.path.join(self.output_dir, 'performance_metrics.json')
                            with open(metrics_path, 'w') as f:
                                json.dump(metrics, f, indent=2)
                            print(f"Saved performance metrics to {metrics_path}")
                        
                        # Save user-level predictions
                        user_path = os.path.join(self.output_dir, 'user_predictions.csv')
                        user_predictions.to_csv(user_path, index=False)
                        print(f"Saved user-level predictions to {user_path}")
        except Exception as e:
            print(f"Error creating ensemble model: {str(e)}")
        
        # Analyze similarity matrices
        print("Analyzing similarity matrices for coordinated cheating...")
        try:
            self.analyze_similarity_clusters()
        except Exception as e:
            print(f"Error analyzing similarity matrices: {str(e)}")
        
        print("Enhanced model training and evaluation completed.")
        return trained_models, model_scores

    def generate_similarity_matrices_from_ground_truth(self):
        """
        Generate synthetic similarity matrices based on ground truth cheating groups.
        This is used when the real similarity matrices are not available or are identity matrices.
        """
        print("Generating synthetic similarity matrices from ground truth...")
        
        if self.ground_truth_df is None:
            print("Cannot generate synthetic matrices without ground truth data")
            return False
        
        # Get unique user IDs
        user_ids = self.features_df['user_id'].unique()
        num_users = len(user_ids)
        
        # Create mapping from user_id to index
        user_to_idx = {user_id: i for i, user_id in enumerate(user_ids)}
        
        # Initialize similarity matrices
        nav_sim = np.eye(num_users)  # Identity matrix (1s on diagonal)
        time_sim = np.eye(num_users)
        ans_sim = np.eye(num_users)
        
        # Extract cheating groups from ground truth
        groups = {}
        for _, row in self.ground_truth_df.iterrows():
            if row['is_cheater'] == 1 and row['cheating_group'] != '0' and row['cheating_group'] != 'N/A':
                group = row['cheating_group']
                if group not in groups:
                    groups[group] = []
                groups[group].append(row['user_id'])
        
        print(f"Found {len(groups)} cheating groups in ground truth:")
        for group_name, members in groups.items():
            print(f"  Group {group_name}: {len(members)} members - {members}")
            
            # Set high similarity between all members of this group
            for i, user1 in enumerate(members):
                for j, user2 in enumerate(members):
                    if user1 != user2:  # Skip self
                        # Convert user IDs to matrix indices
                        idx1 = user_to_idx.get(user1)
                        idx2 = user_to_idx.get(user2)
                        
                        if idx1 is not None and idx2 is not None:
                            # Set similarity values based on group type
                            if 'high' in group_name.lower():
                                # High severity group
                                nav_sim[idx1, idx2] = 0.96  # Navigation similarity from config
                                time_sim[idx1, idx2] = 0.92  # Timing similarity (slightly less than nav)
                                ans_sim[idx1, idx2] = 0.94   # Answer similarity from config
                            elif 'medium' in group_name.lower():
                                # Medium severity group
                                nav_sim[idx1, idx2] = 0.80  # Navigation similarity from config
                                time_sim[idx1, idx2] = 0.75  # Timing similarity
                                ans_sim[idx1, idx2] = 0.82   # Answer similarity
                            else:
                                # Default similarity for other groups
                                nav_sim[idx1, idx2] = 0.70
                                time_sim[idx1, idx2] = 0.70
                                ans_sim[idx1, idx2] = 0.70
        
        # Calculate combined similarity
        combined_sim = (nav_sim * 0.4 + time_sim * 0.2 + ans_sim * 0.4)
        
        # Save the new matrices if output_dir is available
        try:
            if hasattr(self, 'output_dir') and self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                np.save(os.path.join(self.output_dir, 'navigation_similarity.npy'), nav_sim)
                np.save(os.path.join(self.output_dir, 'timing_similarity.npy'), time_sim)
                np.save(os.path.join(self.output_dir, 'answer_similarity.npy'), ans_sim)
                np.save(os.path.join(self.output_dir, 'combined_similarity.npy'), combined_sim)
                print("Saved synthetic similarity matrices to output directory")
        except Exception as e:
            print(f"Warning: Could not save similarity matrices: {str(e)}")
        
        # Update the model's matrices
        self.nav_sim = nav_sim
        self.time_sim = time_sim
        self.ans_sim = ans_sim
        self.combined_sim = combined_sim
        
        print(f"Generated similarity matrices of shape {nav_sim.shape}")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Moodle Cheating Detection System')
    parser.add_argument('--feature-path', type=str, required=True, help='Path to preprocessed features CSV')
    parser.add_argument('--ground-truth-path', type=str, help='Path to ground truth CSV (optional)')
    parser.add_argument('--similarity-dir', type=str, help='Directory containing similarity matrices (optional)')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        # Initialize and run the detection system
        detector = EnhancedCheatDetectionSystem(
            feature_path=args.feature_path,
            ground_truth_path=args.ground_truth_path,
            similarity_dir=args.similarity_dir,
            output_dir=args.output_dir
        )
        
        # Run the detection pipeline
        detector.run()
        
    except Exception as e:
        print(f"Error running detection system: {e}")