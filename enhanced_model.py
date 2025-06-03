# enhanced_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import os

# 1. Model & ML Tooling Imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# --- Configuration & Constants ---

# File Paths
DATA_FILE_PATH = 'data/processed_artificial_features_V2.csv'
OUTPUT_DIR = 'results' # Directory to save models and plots

# Data & Features
TARGET_VARIABLE = 'is_cheater'
FEATURE_COLUMNS = [
    'max_nav_similarity_zscore', 'mean_nav_similarity_zscore', 'median_step_duration', 
    'nav_revisits_count_zscore', 'quick_actions_count', 'std_step_duration', 'sumgrades'
]

# ML Constants
RANDOM_STATE = 42
TEST_SET_SIZE = 0.15
VALIDATION_SET_SIZE = 0.15


# --- Function Definitions ---

def load_and_split_data(file_path, target_col, features):
    """Loads data, scales features, and splits into train, validation, and test sets."""
    print("1. Loading and splitting data...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}. Please check the path.")
    
    df = pd.read_csv(file_path)
    
    X = df[features]
    y = df[target_col]
    
    # First split: (1 - Test_Size) for Train+Val, and Test_Size for Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Adjust validation size relative to the remaining data
    val_size_adjusted = VALIDATION_SET_SIZE / (1 - TEST_SET_SIZE)

    # Second split: Train and Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=RANDOM_STATE, stratify=y_train_val
    )
    
    print(f"Train shape: {X_train.shape} ({len(X_train)/len(df):.0%})")
    print(f"Validation shape: {X_val.shape} ({len(X_val)/len(df):.0%})")
    print(f"Test shape: {X_test.shape} ({len(X_test)/len(df):.0%})")
    
    # Scale features - crucial for SVM and NN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))
    print("Scaler has been saved.")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

def create_neural_network():
    """Creates the scikit-learn MLPClassifier with similar architecture."""
    # Using similar layer sizes as the original TensorFlow model
    # hidden_layer_sizes=(64, 32, 16) creates 3 hidden layers with those sizes
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization similar to dropout
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_STATE
    )
    return model

def tune_and_train_models(X_train, y_train, X_val, y_val):
    """Initializes, tunes, and trains all models."""
    print("\n2. Training and Tuning Models...")
    models = {}

    # --- Hyperparameter Tuning with GridSearchCV ---
    # Justification: These grids explore the trade-off between model complexity,
    # size, and regularization to find a robust model that avoids overfitting.
    
    # RandomForest
    print("Tuning Random Forest...")
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    models['Random Forest'] = rf_grid.best_estimator_
    print(f"Best RF params: {rf_grid.best_params_}")

    # SVM
    print("Tuning SVM...")
    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['rbf']
    }
    svm_grid = GridSearchCV(SVC(probability=True, random_state=RANDOM_STATE), svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    svm_grid.fit(X_train, y_train)
    models['SVM'] = svm_grid.best_estimator_
    print(f"Best SVM params: {svm_grid.best_params_}")

    # --- Train other models ---
    print("Training Gradient Boosting & XGBoost...")
    gb_clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb_clf.fit(X_train, y_train)
    models['Gradient Boosting'] = gb_clf

    xgb_clf = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
    xgb_clf.fit(X_train, y_train)
    models['XGBoost'] = xgb_clf

    # --- Train Neural Network ---
    print("Training Neural Network...")
    nn_model = create_neural_network()
    nn_model.fit(X_train, y_train)
    models['Neural Network'] = nn_model
    
    # --- Create and Train Ensemble Model ---
    print("Training Ensemble Model...")
    # Note: NN is handled differently for voting, so we use a wrapper or custom voting.
    # For simplicity, we'll ensemble the scikit-learn compatible models.
    sklearn_models = {k: v for k, v in models.items() if k != 'Neural Network'}
    
    ensemble_clf = VotingClassifier(
        estimators=[(name, model) for name, model in sklearn_models.items()],
        voting='soft' # Soft voting averages probabilities, usually performs better
    )
    ensemble_clf.fit(X_train, y_train)
    models['Ensemble (Voting)'] = ensemble_clf
    
    print("Model training complete.")
    return models


def evaluate_and_save_models(models, X_test, y_test):
    """Evaluates all models and saves them along with plots and reports."""
    print("\n3. Evaluating models on the test set...")
    report_data = {}

    for name, model in models.items():
        print(f"--- Evaluating {name} ---")
        
        # All models now use the same prediction interface
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Generate and save classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_data[name] = report['accuracy']
        with open(os.path.join(OUTPUT_DIR, f'report_{name}.txt'), 'w') as f:
            f.write(classification_report(y_test, y_pred))
        
        # Generate and save Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.title(f'Confusion Matrix for {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{name}.png'))
        plt.close()

        # Generate and save ROC and Precision-Recall Curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ax2.plot(recall, precision, color='blue', lw=2)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')

        plt.suptitle(f'Evaluation Curves for {name}')
        plt.savefig(os.path.join(OUTPUT_DIR, f'curves_{name}.png'))
        plt.close()

        # Save the model (all models now use joblib)
        dump(model, os.path.join(OUTPUT_DIR, f'model_{name}.joblib'))

    print("\nAll models, plots, and reports have been saved.")

    # --- Identify and highlight the best model ---
    best_model_name = max(report_data, key=report_data.get)
    print(f"\n🏆 Best Model (by accuracy): {best_model_name} with Accuracy: {report_data[best_model_name]:.4f}")
    return best_model_name, models[best_model_name]

def analyze_feature_importance(model, features, model_name):
    """Analyzes and plots feature importance for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        print(f"\nFeature importance not available for {model_name}.")
        return

    print(f"\n4. Analyzing Feature Importance for {model_name}...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importance for {model_name}')
    plt.bar(range(len(features)), importances[indices], align='center')
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'feature_importance_{model_name}.png'))
    plt.close()
    print("Feature importance plot saved.")


def main():
    """Main function to run the entire ML pipeline."""
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Run the pipeline
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(DATA_FILE_PATH, TARGET_VARIABLE, FEATURE_COLUMNS)
    trained_models = tune_and_train_models(X_train, y_train, X_val, y_val)
    best_model_name, best_model = evaluate_and_save_models(trained_models, X_test, y_test)
    
    # Analyze feature importance for tree-based models like RF or XGBoost
    if 'Random Forest' in trained_models:
        analyze_feature_importance(trained_models['Random Forest'], FEATURE_COLUMNS, 'Random Forest')


if __name__ == '__main__':
    main()