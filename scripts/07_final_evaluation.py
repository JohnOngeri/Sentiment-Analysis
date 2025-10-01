"""
Final Model Evaluation and Comparison
Compares all models and generates comprehensive evaluation report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

print("="*70)
print("FINAL MODEL EVALUATION AND COMPARISON")
print("="*70)

# Create results directory
os.makedirs('../results/final_evaluation', exist_ok=True)

# Load all results
print("\n[1/3] Loading all model results...")

# Traditional ML results
with open('../results/traditional_ml/detailed_results.json', 'r') as f:
    trad_ml_results = json.load(f)

# Deep Learning results
with open('../results/deep_learning/detailed_results.json', 'r') as f:
    dl_results = json.load(f)

# Combine all results
all_results = trad_ml_results + dl_results

# Create comprehensive comparison DataFrame
comparison_df = pd.DataFrame(all_results)
comparison_df = comparison_df[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'training_time']]
comparison_df = comparison_df.sort_values('f1_score', ascending=False)

print("\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

# Save comprehensive comparison
comparison_df.to_csv('../results/final_evaluation/all_models_comparison.csv', index=False)

# ===== VISUALIZATIONS =====
print("\n[2/3] Creating comprehensive visualizations...")

# Visualization 1: All Models Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['accuracy', 'precision', 'recall', 'f1_score']
colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_df)))

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    bars = ax.barh(comparison_df['model'], comparison_df[metric], color=colors, alpha=0.8, edgecolor='black')
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Score', fontsize=12)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (v, model) in enumerate(zip(comparison_df[metric], comparison_df['model'])):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('../results/final_evaluation/all_models_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ All models comparison saved")

# Visualization 2: Traditional ML vs Deep Learning
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Categorize models
trad_ml_models = ['Logistic Regression', 'SVM', 'Naive Bayes']
dl_models = ['LSTM', 'Bidirectional LSTM', 'LSTM + MaxPooling']

trad_ml_data = comparison_df[comparison_df['model'].isin(trad_ml_models)]
dl_data = comparison_df[comparison_df['model'].isin(dl_models)]

# Average metrics comparison
avg_metrics = pd.DataFrame({
    'Category': ['Traditional ML', 'Deep Learning'],
    'Accuracy': [trad_ml_data['accuracy'].mean(), dl_data['accuracy'].mean()],
    'Precision': [trad_ml_data['precision'].mean(), dl_data['precision'].mean()],
    'Recall': [trad_ml_data['recall'].mean(), dl_data['recall'].mean()],
    'F1-Score': [trad_ml_data['f1_score'].mean(), dl_data['f1_score'].mean()]
})

# Plot 1: Average metrics
x = np.arange(len(avg_metrics))
width = 0.2
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors_bar = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for i, metric in enumerate(metrics_to_plot):
    offset = width * (i - 1.5)
    axes[0].bar(x + offset, avg_metrics[metric], width, label=metric, color=colors_bar[i], alpha=0.7)

axes[0].set_title('Traditional ML vs Deep Learning (Average Metrics)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_xticks(x)
axes[0].set_xticklabels(avg_metrics['Category'])
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim(0, 1)

# Plot 2: Training time comparison
axes[1].bar(['Traditional ML', 'Deep Learning'], 
            [trad_ml_data['training_time'].mean(), dl_data['training_time'].mean()],
            color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[1].set_title('Average Training Time', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Time (seconds)', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

for i, v in enumerate([trad_ml_data['training_time'].mean(), dl_data['training_time'].mean()]):
    axes[1].text(i, v + 5, f'{v:.1f}s', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/final_evaluation/traditional_vs_deep_learning.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Traditional ML vs Deep Learning comparison saved")

# Visualization 3: Performance vs Training Time Trade-off
plt.figure(figsize=(12, 8))

for idx, row in comparison_df.iterrows():
    if row['model'] in trad_ml_models:
        color = '#3498db'
        marker = 'o'
    else:
        color = '#e74c3c'
        marker = 's'
    
    plt.scatter(row['training_time'], row['f1_score'], 
                s=200, c=color, marker=marker, alpha=0.7, edgecolors='black', linewidth=2)
    plt.annotate(row['model'], (row['training_time'], row['f1_score']), 
                 xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

plt.xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
plt.title('Model Performance vs Training Time Trade-off', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(['Traditional ML', 'Deep Learning'], fontsize=10)

# Add custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', edgecolor='black', label='Traditional ML'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='Deep Learning')
]
plt.legend(handles=legend_elements, fontsize=10)

plt.tight_layout()
plt.savefig('../results/final_evaluation/performance_vs_time_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Performance vs time trade-off saved")

# ===== ANALYSIS AND INSIGHTS =====
print("\n[3/3] Generating analysis and insights...")

best_overall = comparison_df.iloc[0]
best_trad_ml = trad_ml_data.iloc[0]
best_dl = dl_data.iloc[0]

analysis = {
    'best_overall_model': {
        'name': best_overall['model'],
        'accuracy': float(best_overall['accuracy']),
        'f1_score': float(best_overall['f1_score']),
        'training_time': float(best_overall['training_time'])
    },
    'best_traditional_ml': {
        'name': best_trad_ml['model'],
        'accuracy': float(best_trad_ml['accuracy']),
        'f1_score': float(best_trad_ml['f1_score'])
    },
    'best_deep_learning': {
        'name': best_dl['model'],
        'accuracy': float(best_dl['accuracy']),
        'f1_score': float(best_dl['f1_score'])
    },
    'insights': {
        'traditional_ml_avg_accuracy': float(trad_ml_data['accuracy'].mean()),
        'deep_learning_avg_accuracy': float(dl_data['accuracy'].mean()),
        'traditional_ml_avg_time': float(trad_ml_data['training_time'].mean()),
        'deep_learning_avg_time': float(dl_data['training_time'].mean()),
        'performance_difference': float(dl_data['f1_score'].mean() - trad_ml_data['f1_score'].mean()),
        'time_difference': float(dl_data['training_time'].mean() - trad_ml_data['training_time'].mean())
    }
}

# Save analysis
with open('../results/final_evaluation/analysis_summary.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print("\n" + "="*70)
print("FINAL ANALYSIS SUMMARY")
print("="*70)
print(f"\nBest Overall Model: {analysis['best_overall_model']['name']}")
print(f"  - Accuracy: {analysis['best_overall_model']['accuracy']:.4f}")
print(f"  - F1-Score: {analysis['best_overall_model']['f1_score']:.4f}")
print(f"  - Training Time: {analysis['best_overall_model']['training_time']:.2f}s")

print(f"\nBest Traditional ML: {analysis['best_traditional_ml']['name']}")
print(f"  - Accuracy: {analysis['best_traditional_ml']['accuracy']:.4f}")
print(f"  - F1-Score: {analysis['best_traditional_ml']['f1_score']:.4f}")

print(f"\nBest Deep Learning: {analysis['best_deep_learning']['name']}")
print(f"  - Accuracy: {analysis['best_deep_learning']['accuracy']:.4f}")
print(f"  - F1-Score: {analysis['best_deep_learning']['f1_score']:.4f}")

print(f"\nKey Insights:")
print(f"  - Deep Learning avg accuracy: {analysis['insights']['deep_learning_avg_accuracy']:.4f}")
print(f"  - Traditional ML avg accuracy: {analysis['insights']['traditional_ml_avg_accuracy']:.4f}")
print(f"  - Performance difference: {analysis['insights']['performance_difference']:.4f}")
print(f"  - Deep Learning is {analysis['insights']['time_difference']:.1f}s slower on average")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print("1. For Production: Use Logistic Regression (fast, interpretable, good performance)")
print("2. For Best Accuracy: Use LSTM or Bidirectional LSTM (higher performance)")
print("3. For Resource-Constrained: Use Logistic Regression (minimal training time)")
print("4. For Research: Deep Learning models show potential for further improvement")
print("="*70)

print("\n✓ Final evaluation complete!")
print("✓ All results saved in results/final_evaluation/")
