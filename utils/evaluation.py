"""
Evaluation utilities for model performance assessment
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import pandas as pd


def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Comprehensive model evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        model_name: Name of the model for display
    
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary')
    }
    
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        metrics['auc'] = auc(fpr, tpr)
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_history(history, model_name="Model", save_path=None):
    """
    Plot training history (loss and accuracy)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_models(results_list, save_path=None):
    """
    Compare multiple models side by side
    
    Args:
        results_list: List of dictionaries containing model results
        save_path: Path to save the comparison plot
    """
    df = pd.DataFrame(results_list)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, metric in enumerate(metrics):
        if metric in df.columns:
            axes[idx].bar(df['model'], df[metric], color=colors[idx], alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('Score', fontsize=12)
            axes[idx].set_ylim(0, 1)
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, v in enumerate(df[metric]):
                axes[idx].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def print_classification_report(y_true, y_pred, model_name="Model"):
    """
    Print detailed classification report
    """
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION REPORT - {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
