# analyze_top_offenders.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

# Configuration
DETECTION_RESULTS_PATH = 'detection_results/high_confidence_cheaters_Ensemble (Voting)_20250603_141845.csv'
REAL_DATA_PATH = 'data/processed_real_features_for_detection_V2.csv'
TOP_N_OFFENDERS = 5  # Analyze top 5 repeat offenders

# Feature columns for analysis
FEATURE_COLUMNS = [
    'max_nav_similarity_zscore', 'mean_nav_similarity_zscore', 'median_step_duration', 
    'nav_revisits_count', 'quick_actions_count', 'std_nav_similarity_zscore', 
    'std_step_duration', 'sumgrades'
]

# Better feature names for visualization
FEATURE_DISPLAY_NAMES = {
    'max_nav_similarity_zscore': 'Max Navigation Similarity',
    'mean_nav_similarity_zscore': 'Mean Navigation Similarity', 
    'median_step_duration': 'Median Step Duration',
    'nav_revisits_count': 'Navigation Revisits',
    'quick_actions_count': 'Quick Actions Count',
    'std_nav_similarity_zscore': 'Navigation Similarity Variation',
    'std_step_duration': 'Step Duration Variability',
    'sumgrades': 'Total Grade'
}

def load_data():
    """Load detection results and original features."""
    detections = pd.read_csv(DETECTION_RESULTS_PATH)
    real_data = pd.read_csv(REAL_DATA_PATH)
    
    # Merge to get features for detected cheaters
    merged = detections.merge(real_data, on=['attempt_id', 'user_id', 'quiz_id'], how='left')
    
    return detections, real_data, merged

def find_top_offenders(detections):
    """Find users with most cheating detections."""
    user_counts = detections.groupby('user_id').size().sort_values(ascending=False)
    top_users = user_counts.head(TOP_N_OFFENDERS)
    
    print(f"\nTop {TOP_N_OFFENDERS} Repeat Offenders:")
    for user_id, count in top_users.items():
        print(f"User {user_id}: {count} suspicious attempts")
    
    return top_users

def calculate_baseline_stats(real_data):
    """Calculate baseline statistics for comparison."""
    # Load scaler to understand the z-score scaling
    scaler = load('results/scaler.joblib')
    
    # Calculate percentiles for each feature
    baseline_stats = {}
    for feature in FEATURE_COLUMNS:
        baseline_stats[feature] = {
            'mean': real_data[feature].mean(),
            'std': real_data[feature].std(),
            'median': real_data[feature].median(),
            'p25': real_data[feature].quantile(0.25),
            'p75': real_data[feature].quantile(0.75),
            'p95': real_data[feature].quantile(0.95)
        }
    
    return baseline_stats

def create_offender_profile_visualization(user_id, user_data, baseline_stats, real_data):
    """Create detailed visualization for a specific offender."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Cheating Pattern Analysis for User {user_id} ({len(user_data)} detections)', fontsize=16)
    
    # 1. Feature Comparison Radar Chart
    ax1 = plt.subplot(2, 2, 1, projection='polar')
    
    # Calculate average feature values for this user
    user_avg_features = user_data[FEATURE_COLUMNS].mean()
    
    # Normalize features to 0-1 scale for radar chart
    normalized_features = []
    feature_labels = []
    
    for feature in FEATURE_COLUMNS:
        # Use percentile rank for normalization
        value = user_avg_features[feature]
        percentile = (real_data[feature] <= value).mean()
        normalized_features.append(percentile)
        feature_labels.append(FEATURE_DISPLAY_NAMES[feature])
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(FEATURE_COLUMNS), endpoint=False).tolist()
    normalized_features += normalized_features[:1]  # Complete the circle
    angles += angles[:1]
    
    ax1.plot(angles, normalized_features, 'o-', linewidth=2, color='red', label='Cheater Profile')
    ax1.fill(angles, normalized_features, alpha=0.25, color='red')
    ax1.plot(angles, [0.5]*len(angles), '--', color='gray', label='Median User')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(feature_labels, size=8)
    ax1.set_ylim(0, 1)
    ax1.set_title('Feature Profile (Percentile Rank)', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    # 2. Time-based features comparison
    ax2 = plt.subplot(2, 2, 2)
    time_features = ['median_step_duration', 'std_step_duration', 'quick_actions_count']
    
    x = np.arange(len(time_features))
    width = 0.35
    
    user_values = [user_avg_features[f] for f in time_features]
    baseline_values = [baseline_stats[f]['median'] for f in time_features]
    
    bars1 = ax2.bar(x - width/2, user_values, width, label='User', color='red', alpha=0.7)
    bars2 = ax2.bar(x + width/2, baseline_values, width, label='Baseline Median', color='gray', alpha=0.7)
    
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Value')
    ax2.set_title('Time-Based Behavior Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([FEATURE_DISPLAY_NAMES[f] for f in time_features], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Navigation similarity patterns
    ax3 = plt.subplot(2, 2, 3)
    nav_features = ['max_nav_similarity_zscore', 'mean_nav_similarity_zscore', 'std_nav_similarity_zscore']
    
    # Show distribution of these features for the user
    user_nav_data = [user_data[f].values for f in nav_features]
    
    ax3.boxplot(user_nav_data, tick_labels=[FEATURE_DISPLAY_NAMES[f] for f in nav_features])
    ax3.set_title('Navigation Pattern Distribution Across Attempts')
    ax3.set_ylabel('Z-Score Value')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Population Mean')
    ax3.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='2 SD Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Rotate x labels
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Cheating probability over attempts
    ax4 = plt.subplot(2, 2, 4)
    
    # Sort by cheating probability to show pattern
    sorted_data = user_data.sort_values('cheating_probability', ascending=False)
    attempts = range(len(sorted_data))
    
    ax4.scatter(attempts, sorted_data['cheating_probability'], c='red', alpha=0.6, s=50)
    ax4.axhline(y=0.8, color='red', linestyle='--', label='High Confidence Threshold')
    ax4.set_xlabel('Attempt Number (sorted by probability)')
    ax4.set_ylabel('Cheating Probability')
    ax4.set_title('Cheating Probability Distribution')
    ax4.set_ylim(0.75, 1.0)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'detection_results/offender_analysis_user_{user_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_comparison():
    """Create a summary comparison of top offenders vs normal users."""
    detections, real_data, merged = load_data()
    
    # Get all cheater data
    cheater_features = merged[FEATURE_COLUMNS]
    
    # Sample normal users (those not in detections)
    normal_users = real_data[~real_data['user_id'].isin(detections['user_id'].unique())]
    normal_features = normal_users[FEATURE_COLUMNS].sample(n=min(1000, len(normal_users)))
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Feature Distribution: Detected Cheaters vs Normal Users', fontsize=16)
    
    for idx, (feature, ax) in enumerate(zip(FEATURE_COLUMNS, axes.flatten())):
        # Create violin plots
        data_to_plot = [normal_features[feature].values, cheater_features[feature].values]
        
        parts = ax.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)
        
        # Customize colors
        for pc, color in zip(parts['bodies'], ['lightblue', 'lightcoral']):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Normal', 'Cheater'])
        ax.set_title(FEATURE_DISPLAY_NAMES[feature])
        ax.grid(True, alpha=0.3)
        
        # Add statistical annotation
        normal_mean = normal_features[feature].mean()
        cheater_mean = cheater_features[feature].mean()
        
        ax.text(0.02, 0.98, f'Δ = {cheater_mean - normal_mean:.2f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove extra subplot
    fig.delaxes(axes[1, 3])
    
    plt.tight_layout()
    plt.savefig('detection_results/cheater_vs_normal_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis function."""
    print("Loading detection data...")
    detections, real_data, merged = load_data()
    
    print("\nCalculating baseline statistics...")
    baseline_stats = calculate_baseline_stats(real_data)
    
    # Find top offenders
    top_offenders = find_top_offenders(detections)
    
    # Create detailed analysis for each top offender
    print("\nGenerating detailed visualizations for top offenders...")
    for user_id in top_offenders.index[:TOP_N_OFFENDERS]:
        user_data = merged[merged['user_id'] == user_id]
        create_offender_profile_visualization(user_id, user_data, baseline_stats, real_data)
        print(f"✓ Created analysis for User {user_id}")
    
    # Create summary comparison
    print("\nCreating summary comparison visualization...")
    create_summary_comparison()
    
    print("\n✅ Analysis complete! Check detection_results/ for visualizations.")
    
    # Print key insights
    print("\n=== Key Insights ===")
    for user_id in top_offenders.index[:3]:  # Top 3 offenders
        user_data = merged[merged['user_id'] == user_id]
        print(f"\nUser {user_id} ({len(user_data)} detections):")
        print(f"  - Avg Navigation Similarity: {user_data['mean_nav_similarity_zscore'].mean():.2f} SD above mean")
        print(f"  - Quick Actions: {user_data['quick_actions_count'].mean():.0f} per attempt")
        print(f"  - Median Step Duration: {user_data['median_step_duration'].mean():.1f} seconds")

if __name__ == '__main__':
    main() 