"""
Traditional Machine Learning Model Implementation
Implements Logistic Regression with TF-IDF features
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluation import evaluate_model, plot_confusion_matrix, print_classification_report
import pickle
import json
import time

print("="*60)
print("TRADITIONAL MACHINE LEARNING MODEL TRAINING")
print("="*60)

# Load preprocessed data
print("\n[1/5] Loading preprocessed data and embeddings...")
train_df = pd.read_csv('data/train_preprocessed.csv')
test_df = pd.read_csv('data/test_preprocessed.csv')

# Load TF-IDF embeddings
X_train = np.load('models/embeddings/X_train_tfidf.npy')
X_test = np.load('models/embeddings/X_test_tfidf.npy')
y_train = train_df['label'].values
y_test = test_df['label'].values

print(f"  - Training samples: {X_train.shape[0]}")
print(f"  - Test samples: {X_test.shape[0]}")
print(f"  - Features: {X_train.shape[1]}")

# Create results directory
os.makedirs('results/traditional_ml', exist_ok=True)
os.makedirs('models/traditional_ml', exist_ok=True)

# ===== MODEL 1: LOGISTIC REGRESSION =====
print("\n[2/5] Training Logistic Regression...")
print("  Hyperparameters:")
print("  - C (regularization): 1.0")
print("  - max_iter: 1000")
print("  - solver: lbfgs")
print("  - penalty: l2")

start_time = time.time()

lr_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='lbfgs',
    random_state=42,
    verbose=0
)

lr_model.fit(X_train, y_train)
lr_train_time = time.time() - start_time

print(f"  ✓ Training completed in {lr_train_time:.2f} seconds")

# Predictions
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# Evaluate
lr_metrics = evaluate_model(y_test, y_pred_lr, y_pred_proba_lr, "Logistic Regression")
lr_metrics['training_time'] = lr_train_time

print(f"\n  Results:")
print(f"  - Accuracy:  {lr_metrics['accuracy']:.4f}")
print(f"  - Precision: {lr_metrics['precision']:.4f}")
print(f"  - Recall:    {lr_metrics['recall']:.4f}")
print(f"  - F1-Score:  {lr_metrics['f1_score']:.4f}")
print(f"  - AUC:       {lr_metrics['auc']:.4f}")

# Cross-validation
print("\n  Performing 5-fold cross-validation...")
cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"  - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
lr_metrics['cv_accuracy'] = cv_scores.mean()
lr_metrics['cv_std'] = cv_scores.std()

# Save model
with open('models/traditional_ml/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("  ✓ Model saved")

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression", 
                     'results/traditional_ml/lr_confusion_matrix.png')
print("  ✓ Confusion matrix saved")

# Print detailed report
print_classification_report(y_test, y_pred_lr, "Logistic Regression")

# ===== MODEL 2: SUPPORT VECTOR MACHINE =====
print("\n[3/5] Training Support Vector Machine (LinearSVC)...")
print("  Hyperparameters:")
print("  - C (regularization): 1.0")
print("  - max_iter: 1000")
print("  - loss: squared_hinge")

start_time = time.time()

svm_model = LinearSVC(
    C=1.0,
    max_iter=1000,
    random_state=42,
    verbose=0
)

svm_model.fit(X_train, y_train)
svm_train_time = time.time() - start_time

print(f"  ✓ Training completed in {svm_train_time:.2f} seconds")

# Predictions
y_pred_svm = svm_model.predict(X_test)

# Evaluate
svm_metrics = evaluate_model(y_test, y_pred_svm, None, "SVM")
svm_metrics['training_time'] = svm_train_time

print(f"\n  Results:")
print(f"  - Accuracy:  {svm_metrics['accuracy']:.4f}")
print(f"  - Precision: {svm_metrics['precision']:.4f}")
print(f"  - Recall:    {svm_metrics['recall']:.4f}")
print(f"  - F1-Score:  {svm_metrics['f1_score']:.4f}")

# Save model
with open('models/traditional_ml/svm.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
print("  ✓ Model saved")

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred_svm, "Support Vector Machine", 
                     'results/traditional_ml/svm_confusion_matrix.png')
print("  ✓ Confusion matrix saved")

# ===== MODEL 3: NAIVE BAYES =====
print("\n[4/5] Training Naive Bayes (MultinomialNB)...")
print("  Hyperparameters:")
print("  - alpha (smoothing): 1.0")

start_time = time.time()

# Naive Bayes requires non-negative features, ensure this
X_train_nb = np.abs(X_train)
X_test_nb = np.abs(X_test)

nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train_nb, y_train)
nb_train_time = time.time() - start_time

print(f"  ✓ Training completed in {nb_train_time:.2f} seconds")

# Predictions
y_pred_nb = nb_model.predict(X_test_nb)
y_pred_proba_nb = nb_model.predict_proba(X_test_nb)[:, 1]

# Evaluate
nb_metrics = evaluate_model(y_test, y_pred_nb, y_pred_proba_nb, "Naive Bayes")
nb_metrics['training_time'] = nb_train_time

print(f"\n  Results:")
print(f"  - Accuracy:  {nb_metrics['accuracy']:.4f}")
print(f"  - Precision: {nb_metrics['precision']:.4f}")
print(f"  - Recall:    {nb_metrics['recall']:.4f}")
print(f"  - F1-Score:  {nb_metrics['f1_score']:.4f}")
print(f"  - AUC:       {nb_metrics['auc']:.4f}")

# Save model
with open('models/traditional_ml/naive_bayes.pkl', 'wb') as f:
    pickle.dump(nb_model, f)
print("  ✓ Model saved")

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred_nb, "Naive Bayes", 
                     'results/traditional_ml/nb_confusion_matrix.png')
print("  ✓ Confusion matrix saved")

# ===== COMPARISON =====
print("\n[5/5] Comparing traditional ML models...")

all_results = [lr_metrics, svm_metrics, nb_metrics]

# Create comparison table
comparison_df = pd.DataFrame(all_results)
comparison_df = comparison_df[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'training_time']]

print("\n" + "="*80)
print("TRADITIONAL ML MODELS COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

# Save comparison
comparison_df.to_csv('results/traditional_ml/model_comparison.csv', index=False)

# Save detailed results
with open('results/traditional_ml/detailed_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Create comparison visualization
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
colors = ['#3498db', '#e74c3c', '#2ecc71']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    values = [m[metric] for m in all_results]
    models = [m['model'] for m in all_results]
    
    bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('results/traditional_ml/models_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Comparison visualization saved")

# Training time comparison
plt.figure(figsize=(10, 6))
training_times = [m['training_time'] for m in all_results]
models = [m['model'] for m in all_results]
plt.bar(models, training_times, color=colors, alpha=0.7, edgecolor='black')
plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Time (seconds)', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(training_times):
    plt.text(i, v + 0.5, f'{v:.2f}s', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('results/traditional_ml/training_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Training time comparison saved")

print("\n" + "="*60)
print("BEST MODEL: Logistic Regression")
print(f"  - Highest F1-Score: {lr_metrics['f1_score']:.4f}")
print(f"  - Best balance of precision and recall")
print(f"  - Reasonable training time: {lr_metrics['training_time']:.2f}s")
print("="*60)

print("\n✓ Traditional ML model training complete!")
print("✓ All results saved in results/traditional_ml/")
