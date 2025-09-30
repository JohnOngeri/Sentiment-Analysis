"""
Comprehensive Experiments and Evaluation
Tests different hyperparameters, embeddings, and configurations
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluation import evaluate_model
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

print("="*70)
print("COMPREHENSIVE EXPERIMENTS AND HYPERPARAMETER TUNING")
print("="*70)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Create results directory
os.makedirs('results/experiments', exist_ok=True)

# Load data
print("\n[1/3] Loading preprocessed data...")
train_df = pd.read_csv('data/train_preprocessed.csv')
test_df = pd.read_csv('data/test_preprocessed.csv')

# Load embeddings
X_train_tfidf = np.load('models/embeddings/X_train_tfidf.npy')
X_test_tfidf = np.load('models/embeddings/X_test_tfidf.npy')
X_train_w2v = np.load('models/embeddings/X_train_w2v.npy')
X_test_w2v = np.load('models/embeddings/X_test_w2v.npy')

y_train = train_df['label'].values
y_test = test_df['label'].values

# Load tokenizer for deep learning
with open('models/deep_learning/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

X_train_text = train_df['cleaned_text'].values
X_test_text = test_df['cleaned_text'].values

# ===== EXPERIMENT TABLE 1: EMBEDDING COMPARISON =====
print("\n[2/3] EXPERIMENT TABLE 1: Embedding Comparison")
print("-" * 70)

experiment1_results = []

# Test 1: TF-IDF with Logistic Regression
print("\n  Test 1.1: TF-IDF + Logistic Regression")
start_time = time.time()
lr_tfidf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_tfidf.fit(X_train_tfidf, y_train)
y_pred = lr_tfidf.predict(X_test_tfidf)
y_pred_proba = lr_tfidf.predict_proba(X_test_tfidf)[:, 1]
train_time = time.time() - start_time

metrics = evaluate_model(y_test, y_pred, y_pred_proba, "LR + TF-IDF")
experiment1_results.append({
    'Embedding': 'TF-IDF',
    'Model': 'Logistic Regression',
    'Accuracy': metrics['accuracy'],
    'Precision': metrics['precision'],
    'Recall': metrics['recall'],
    'F1-Score': metrics['f1_score'],
    'AUC': metrics['auc'],
    'Training Time (s)': train_time
})
print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

# Test 2: Word2Vec with Logistic Regression
print("\n  Test 1.2: Word2Vec + Logistic Regression")
start_time = time.time()
lr_w2v = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_w2v.fit(X_train_w2v, y_train)
y_pred = lr_w2v.predict(X_test_w2v)
y_pred_proba = lr_w2v.predict_proba(X_test_w2v)[:, 1]
train_time = time.time() - start_time

metrics = evaluate_model(y_test, y_pred, y_pred_proba, "LR + Word2Vec")
experiment1_results.append({
    'Embedding': 'Word2Vec',
    'Model': 'Logistic Regression',
    'Accuracy': metrics['accuracy'],
    'Precision': metrics['precision'],
    'Recall': metrics['recall'],
    'F1-Score': metrics['f1_score'],
    'AUC': metrics['auc'],
    'Training Time (s)': train_time
})
print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

# Test 3: Word2Vec with LSTM
print("\n  Test 1.3: Word2Vec + LSTM")
from gensim.models import Word2Vec
w2v_model = Word2Vec.load('models/embeddings/word2vec.model')

MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 100

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# Create embedding matrix
embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i >= MAX_WORDS:
        continue
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

start_time = time.time()
model_lstm_w2v = Sequential([
    Embedding(MAX_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], 
              input_length=MAX_SEQUENCE_LENGTH, trainable=True),
    SpatialDropout1D(0.2),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model_lstm_w2v.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_lstm_w2v.fit(
    X_train_pad, y_train,
    batch_size=64, epochs=5,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
    verbose=0
)
train_time = time.time() - start_time

y_pred_proba = model_lstm_w2v.predict(X_test_pad, verbose=0).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

metrics = evaluate_model(y_test, y_pred, y_pred_proba, "LSTM + Word2Vec")
experiment1_results.append({
    'Embedding': 'Word2Vec',
    'Model': 'LSTM',
    'Accuracy': metrics['accuracy'],
    'Precision': metrics['precision'],
    'Recall': metrics['recall'],
    'F1-Score': metrics['f1_score'],
    'AUC': metrics['auc'],
    'Training Time (s)': train_time
})
print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

# Save Experiment 1 results
exp1_df = pd.DataFrame(experiment1_results)
exp1_df.to_csv('results/experiments/experiment1_embedding_comparison.csv', index=False)

print("\n" + "="*70)
print("EXPERIMENT 1 RESULTS: Embedding Comparison")
print("="*70)
print(exp1_df.to_string(index=False))
print("="*70)

# ===== EXPERIMENT TABLE 2: HYPERPARAMETER TUNING =====
print("\n[3/3] EXPERIMENT TABLE 2: Hyperparameter Tuning (LSTM)")
print("-" * 70)

experiment2_results = []

# Define hyperparameter configurations
configs = [
    {'learning_rate': 0.001, 'batch_size': 32, 'optimizer': 'adam', 'lstm_units': 128},
    {'learning_rate': 0.001, 'batch_size': 64, 'optimizer': 'adam', 'lstm_units': 128},
    {'learning_rate': 0.001, 'batch_size': 128, 'optimizer': 'adam', 'lstm_units': 128},
    {'learning_rate': 0.0001, 'batch_size': 64, 'optimizer': 'adam', 'lstm_units': 128},
    {'learning_rate': 0.01, 'batch_size': 64, 'optimizer': 'adam', 'lstm_units': 128},
    {'learning_rate': 0.001, 'batch_size': 64, 'optimizer': 'rmsprop', 'lstm_units': 128},
    {'learning_rate': 0.001, 'batch_size': 64, 'optimizer': 'adam', 'lstm_units': 64},
    {'learning_rate': 0.001, 'batch_size': 64, 'optimizer': 'adam', 'lstm_units': 256},
]

for idx, config in enumerate(configs, 1):
    print(f"\n  Test 2.{idx}: LR={config['learning_rate']}, BS={config['batch_size']}, "
          f"Opt={config['optimizer']}, Units={config['lstm_units']}")
    
    # Build model
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], 
                  input_length=MAX_SEQUENCE_LENGTH, trainable=True),
        SpatialDropout1D(0.2),
        LSTM(config['lstm_units'], dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile with specified optimizer and learning rate
    if config['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=config['learning_rate'])
    else:
        optimizer = config['optimizer']
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train
    start_time = time.time()
    history = model.fit(
        X_train_pad, y_train,
        batch_size=config['batch_size'],
        epochs=5,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
        verbose=0
    )
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred_proba = model.predict(X_test_pad, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, f"Config {idx}")
    
    # Get final training and validation metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    experiment2_results.append({
        'Learning Rate': config['learning_rate'],
        'Batch Size': config['batch_size'],
        'Optimizer': config['optimizer'],
        'LSTM Units': config['lstm_units'],
        'Accuracy': metrics['accuracy'],
        'F1-Score': metrics['f1_score'],
        'Train Loss': final_train_loss,
        'Val Loss': final_val_loss,
        'Training Time (s)': train_time
    })
    
    print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, "
          f"Val Loss: {final_val_loss:.4f}")

# Save Experiment 2 results
exp2_df = pd.DataFrame(experiment2_results)
exp2_df.to_csv('results/experiments/experiment2_hyperparameter_tuning.csv', index=False)

print("\n" + "="*70)
print("EXPERIMENT 2 RESULTS: Hyperparameter Tuning")
print("="*70)
print(exp2_df.to_string(index=False))
print("="*70)

# ===== VISUALIZATIONS =====
print("\n[4/4] Creating experiment visualizations...")

# Visualization 1: Embedding Comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

metrics_exp1 = ['Accuracy', 'F1-Score', 'AUC', 'Training Time (s)']
for idx, metric in enumerate(metrics_exp1):
    ax = axes[idx // 2, idx % 2]
    
    # Group by embedding
    for embedding in exp1_df['Embedding'].unique():
        subset = exp1_df[exp1_df['Embedding'] == embedding]
        x_pos = np.arange(len(subset))
        ax.bar(x_pos + (0.35 if embedding == 'Word2Vec' else 0), 
               subset[metric].values, 
               width=0.35, 
               label=embedding,
               alpha=0.7)
    
    ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xticks(np.arange(len(exp1_df) // 2) + 0.175)
    ax.set_xticklabels(exp1_df['Model'].unique(), rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/experiments/experiment1_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Experiment 1 visualization saved")

# Visualization 2: Hyperparameter Effects
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Learning Rate Effect
ax = axes[0, 0]
lr_data = exp2_df[exp2_df['Batch Size'] == 64].sort_values('Learning Rate')
ax.plot(lr_data['Learning Rate'], lr_data['Accuracy'], marker='o', linewidth=2, markersize=8)
ax.set_xscale('log')
ax.set_title('Learning Rate vs Accuracy', fontsize=14, fontweight='bold')
ax.set_xlabel('Learning Rate', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.grid(True, alpha=0.3)

# Batch Size Effect
ax = axes[0, 1]
bs_data = exp2_df[(exp2_df['Learning Rate'] == 0.001) & (exp2_df['Optimizer'] == 'adam') & (exp2_df['LSTM Units'] == 128)]
ax.bar(bs_data['Batch Size'].astype(str), bs_data['Accuracy'], color='#3498db', alpha=0.7)
ax.set_title('Batch Size vs Accuracy', fontsize=14, fontweight='bold')
ax.set_xlabel('Batch Size', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Optimizer Comparison
ax = axes[1, 0]
opt_data = exp2_df[(exp2_df['Learning Rate'] == 0.001) & (exp2_df['Batch Size'] == 64) & (exp2_df['LSTM Units'] == 128)]
ax.bar(opt_data['Optimizer'], opt_data['F1-Score'], color=['#e74c3c', '#2ecc71'], alpha=0.7)
ax.set_title('Optimizer Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Optimizer', fontsize=12)
ax.set_ylabel('F1-Score', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# LSTM Units Effect
ax = axes[1, 1]
units_data = exp2_df[(exp2_df['Learning Rate'] == 0.001) & (exp2_df['Batch Size'] == 64) & (exp2_df['Optimizer'] == 'adam')]
ax.bar(units_data['LSTM Units'].astype(str), units_data['Accuracy'], color='#f39c12', alpha=0.7)
ax.set_title('LSTM Units vs Accuracy', fontsize=14, fontweight='bold')
ax.set_xlabel('LSTM Units', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/experiments/experiment2_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Experiment 2 visualization saved")

# Best configuration summary
best_config = exp2_df.loc[exp2_df['F1-Score'].idxmax()]

print("\n" + "="*70)
print("BEST HYPERPARAMETER CONFIGURATION")
print("="*70)
print(f"Learning Rate: {best_config['Learning Rate']}")
print(f"Batch Size: {best_config['Batch Size']}")
print(f"Optimizer: {best_config['Optimizer']}")
print(f"LSTM Units: {best_config['LSTM Units']}")
print(f"Accuracy: {best_config['Accuracy']:.4f}")
print(f"F1-Score: {best_config['F1-Score']:.4f}")
print("="*70)

print("\n✓ All experiments completed successfully!")
print("✓ Results saved in results/experiments/")
