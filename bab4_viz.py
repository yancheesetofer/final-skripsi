import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load detection results
print("Loading detection results...")
detection_df = pd.read_csv('detection_results/all_detections_Ensemble (Voting)_20250603_141845.csv')
high_conf_df = pd.read_csv('detection_results/high_confidence_cheaters_Ensemble (Voting)_20250603_141845.csv')

# Load training data for correlation analysis
train_df = pd.read_csv('data/processed_artificial_features_V2.csv')

# 1. Distribution of Cheating Probability Scores
print("Creating probability distribution visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Histogram
ax1.hist(detection_df['cheating_probability'], bins=50, alpha=0.7, edgecolor='black')
ax1.axvline(0.8, color='red', linestyle='--', linewidth=2, label='High Confidence Threshold (0.8)')
ax1.axvline(0.6, color='orange', linestyle='--', linewidth=2, label='Medium Confidence Threshold (0.6)')
ax1.axvline(0.4, color='yellow', linestyle='--', linewidth=2, label='Low Confidence Threshold (0.4)')
ax1.set_xlabel('Probabilitas Kecurangan', fontsize=12)
ax1.set_ylabel('Jumlah Percobaan Ujian', fontsize=12)
ax1.set_title('Distribusi Probabilitas Kecurangan pada Data Riil', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Density plot
ax2.plot(sorted(detection_df['cheating_probability']), 
         np.arange(1, len(detection_df) + 1) / len(detection_df),
         linewidth=2)
ax2.fill_between(sorted(detection_df['cheating_probability']), 
                 0, 
                 np.arange(1, len(detection_df) + 1) / len(detection_df),
                 alpha=0.3)
ax2.axvline(0.8, color='red', linestyle='--', linewidth=2)
ax2.axvline(0.6, color='orange', linestyle='--', linewidth=2)
ax2.axvline(0.4, color='yellow', linestyle='--', linewidth=2)
ax2.set_xlabel('Probabilitas Kecurangan', fontsize=12)
ax2.set_ylabel('Probabilitas Kumulatif', fontsize=12)
ax2.set_title('Fungsi Distribusi Kumulatif Probabilitas Kecurangan', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('draft-skripsi-tex/figures/probability_distribution_analysis.pdf', bbox_inches='tight')
plt.close()

# 2. Analysis by Quiz (Top 10 most problematic quizzes)
print("Creating quiz analysis visualization...")
quiz_stats = detection_df.groupby('quiz_id').agg({
    'high_confidence_cheater': ['sum', 'count', 'mean'],
    'cheating_probability': ['mean', 'std']
}).round(4)

quiz_stats.columns = ['Deteksi_High_Conf', 'Total_Attempts', 'Rate_High_Conf', 'Mean_Prob', 'Std_Prob']
quiz_stats = quiz_stats.sort_values('Rate_High_Conf', ascending=False).head(15)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Bar plot of cheating rates by quiz
quiz_stats['Rate_High_Conf'].plot(kind='bar', ax=ax1, color='darkred', alpha=0.7)
ax1.set_xlabel('Quiz ID', fontsize=12)
ax1.set_ylabel('Tingkat Deteksi High Confidence', fontsize=12)
ax1.set_title('15 Ujian dengan Tingkat Kecurangan Tertinggi', fontsize=14)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

# Scatter plot: Total attempts vs detection rate
ax2.scatter(quiz_stats['Total_Attempts'], quiz_stats['Rate_High_Conf'], 
           s=100, alpha=0.6, c=quiz_stats['Mean_Prob'], cmap='Reds')
ax2.set_xlabel('Jumlah Percobaan Ujian', fontsize=12)
ax2.set_ylabel('Tingkat Deteksi High Confidence', fontsize=12)
ax2.set_title('Hubungan Jumlah Peserta dengan Tingkat Kecurangan', fontsize=14)
ax2.grid(True, alpha=0.3)
cbar = plt.colorbar(ax2.collections[0], ax=ax2)
cbar.set_label('Mean Cheating Probability', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig('draft-skripsi-tex/figures/quiz_analysis.pdf', bbox_inches='tight')
plt.close()

# 3. Feature Correlation Heatmap
print("Creating feature correlation heatmap...")
feature_cols = [
    'max_nav_similarity_zscore', 'mean_nav_similarity_zscore', 'median_step_duration', 
    'nav_revisits_count', 'quick_actions_count', 'std_nav_similarity_zscore', 
    'std_step_duration', 'sumgrades'
]

correlation_matrix = train_df[feature_cols].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            fmt='.3f', 
            cmap='coolwarm', 
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": .8})
plt.title('Matriks Korelasi Antar Fitur', fontsize=14)
plt.tight_layout()
plt.savefig('draft-skripsi-tex/figures/feature_correlation_heatmap.pdf', bbox_inches='tight')
plt.close()

# 4. Repeat Offender Analysis
print("Creating repeat offender analysis...")
user_counts = high_conf_df['user_id'].value_counts()
repeat_offenders = user_counts[user_counts > 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Distribution of detection counts
detection_distribution = repeat_offenders.value_counts().sort_index()
ax1.bar(detection_distribution.index, detection_distribution.values, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Jumlah Deteksi per User', fontsize=12)
ax1.set_ylabel('Jumlah User', fontsize=12)
ax1.set_title('Distribusi Repeat Offenders', fontsize=14)
ax1.grid(True, alpha=0.3, axis='y')

# Top 20 repeat offenders
top_offenders = repeat_offenders.head(20)
ax2.barh(range(len(top_offenders)), top_offenders.values, alpha=0.7)
ax2.set_yticks(range(len(top_offenders)))
ax2.set_yticklabels([f'User {uid}' for uid in top_offenders.index])
ax2.set_xlabel('Jumlah Deteksi', fontsize=12)
ax2.set_title('20 User dengan Deteksi Terbanyak', fontsize=14)
ax2.grid(True, alpha=0.3, axis='x')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('draft-skripsi-tex/figures/repeat_offender_analysis.pdf', bbox_inches='tight')
plt.close()

# 5. Model Performance Comparison Visualization
print("Creating model performance comparison...")
models_performance = {
    'Random Forest': {'accuracy': 0.98, 'precision': 1.00, 'recall': 0.93, 'f1': 0.97},
    'SVM': {'accuracy': 0.98, 'precision': 1.00, 'recall': 0.93, 'f1': 0.97},
    'Neural Network': {'accuracy': 0.97, 'precision': 1.00, 'recall': 0.90, 'f1': 0.95},
    'Ensemble (Voting)': {'accuracy': 0.97, 'precision': 0.97, 'recall': 0.97, 'f1': 0.97},
    'XGBoost': {'accuracy': 0.96, 'precision': 0.96, 'recall': 0.93, 'f1': 0.94},
    'Gradient Boosting': {'accuracy': 0.95, 'precision': 0.95, 'recall': 0.90, 'f1': 0.92}
}

metrics_df = pd.DataFrame(models_performance).T
metrics_df = metrics_df.sort_values('f1', ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(metrics_df))
width = 0.2

bars1 = ax.bar(x - 1.5*width, metrics_df['accuracy'], width, label='Accuracy', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, metrics_df['precision'], width, label='Precision', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, metrics_df['recall'], width, label='Recall', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, metrics_df['f1'], width, label='F1-Score', alpha=0.8)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Perbandingan Kinerja Model Machine Learning', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

for bars in [bars1, bars2, bars3, bars4]:
    autolabel(bars)

plt.tight_layout()
plt.savefig('draft-skripsi-tex/figures/model_performance_comparison.pdf', bbox_inches='tight')
plt.close()

print("\nAll visualizations have been created successfully!")
print("Files saved in draft-skripsi-tex/figures/")