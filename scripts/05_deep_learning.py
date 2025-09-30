"""
Deep Learning Model Implementation
Implements LSTM with word embeddings for sentiment analysis
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Bidirectional, 
    SpatialDropout1D, GlobalMaxPooling1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluation import evaluate_model, plot_confusion_matrix, plot_training_history, print_classification_report
import pickle
import json
import time
import matplotlib.pyplot as plt

print("="*60)
print("DEEP LEARNING MODEL TRAINING (LSTM)")
print("="*60)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load preprocessed data
print("\n[1/7] Loading preprocessed data...")
train_df = pd.read_csv('data/train_preprocessed.csv')
test_df = pd.read_csv('data/test_preprocessed.csv')

X_train_text = train_df['cleaned_text'].values
X_test_text = test_df['cleaned_text'].values
y_train = train_df['label'].values
y_test = test_df['label'].values

print(f"  - Training samples: {len(X_train_text)}")
print(f"  - Test samples: {len(X_test_text)}")

# Create results directory
os.makedirs('results/deep_learning', exist_ok=True)
os.makedirs('models/deep_learning', exist_ok=True)

# ===== TOKENIZATION AND PADDING =====
print("\n[2/7] Tokenizing and padding sequences...")

# Hyperparameters
MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train_text)

# Convert texts to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

print(f"  - Vocabulary size: {len(tokenizer.word_index)}")
print(f"  - Max words: {MAX_WORDS}")
print(f"  - Sequence length: {MAX_SEQUENCE_LENGTH}")
print(f"  - Training shape: {X_train_pad.shape}")
print(f"  - Test shape: {X_test_pad.shape}")

# Save tokenizer
with open('models/deep_learning/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("  ✓ Tokenizer saved")

# ===== LOAD WORD2VEC EMBEDDINGS =====
print("\n[3/7] Creating embedding matrix from Word2Vec...")

from gensim.models import Word2Vec
w2v_model = Word2Vec.load('models/embeddings/word2vec.model')

# Create embedding matrix
embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM))
word_index = tokenizer.word_index

for word, i in word_index.items():
    if i >= MAX_WORDS:
        continue
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

print(f"  - Embedding matrix shape: {embedding_matrix.shape}")
print(f"  - Words with embeddings: {np.sum(np.any(embedding_matrix != 0, axis=1))}/{MAX_WORDS}")

# ===== MODEL 1: BASIC LSTM =====
print("\n[4/7] Building and training Basic LSTM model...")

print("\n  Architecture:")
print("  - Embedding Layer (trainable)")
print("  - LSTM Layer (128 units)")
print("  - Dropout (0.5)")
print("  - Dense Output (sigmoid)")

model_lstm = Sequential([
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=True
    ),
    SpatialDropout1D(0.2),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_lstm.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n  Model Summary:")
model_lstm.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint('models/deep_learning/lstm_best.keras', save_best_only=True, monitor='val_accuracy', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001, verbose=1)
]

# Train model
print("\n  Training...")
start_time = time.time()

history_lstm = model_lstm.fit(
    X_train_pad, y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

lstm_train_time = time.time() - start_time
print(f"\n  ✓ Training completed in {lstm_train_time:.2f} seconds")

# Evaluate
print("\n  Evaluating on test set...")
y_pred_proba_lstm = model_lstm.predict(X_test_pad, verbose=0).flatten()
y_pred_lstm = (y_pred_proba_lstm > 0.5).astype(int)

lstm_metrics = evaluate_model(y_test, y_pred_lstm, y_pred_proba_lstm, "LSTM")
lstm_metrics['training_time'] = lstm_train_time

print(f"\n  Results:")
print(f"  - Accuracy:  {lstm_metrics['accuracy']:.4f}")
print(f"  - Precision: {lstm_metrics['precision']:.4f}")
print(f"  - Recall:    {lstm_metrics['recall']:.4f}")
print(f"  - F1-Score:  {lstm_metrics['f1_score']:.4f}")
print(f"  - AUC:       {lstm_metrics['auc']:.4f}")

# Save model
model_lstm.save('models/deep_learning/lstm_final.keras')
print("  ✓ Model saved")

# Plot training history
history_dict = {
    'accuracy': history_lstm.history['accuracy'],
    'val_accuracy': history_lstm.history['val_accuracy'],
    'loss': history_lstm.history['loss'],
    'val_loss': history_lstm.history['val_loss']
}

plot_training_history(history_dict, "LSTM", 'results/deep_learning/lstm_training_history.png')
print("  ✓ Training history saved")

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred_lstm, "LSTM", 'results/deep_learning/lstm_confusion_matrix.png')
print("  ✓ Confusion matrix saved")

# Print detailed report
print_classification_report(y_test, y_pred_lstm, "LSTM")

# ===== MODEL 2: BIDIRECTIONAL LSTM =====
print("\n[5/7] Building and training Bidirectional LSTM model...")

print("\n  Architecture:")
print("  - Embedding Layer (trainable)")
print("  - Bidirectional LSTM (64 units)")
print("  - Dropout (0.5)")
print("  - Dense Output (sigmoid)")

model_bilstm = Sequential([
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=True
    ),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_bilstm.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
print("\n  Training...")
start_time = time.time()

history_bilstm = model_bilstm.fit(
    X_train_pad, y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

bilstm_train_time = time.time() - start_time
print(f"\n  ✓ Training completed in {bilstm_train_time:.2f} seconds")

# Evaluate
y_pred_proba_bilstm = model_bilstm.predict(X_test_pad, verbose=0).flatten()
y_pred_bilstm = (y_pred_proba_bilstm > 0.5).astype(int)

bilstm_metrics = evaluate_model(y_test, y_pred_bilstm, y_pred_proba_bilstm, "Bidirectional LSTM")
bilstm_metrics['training_time'] = bilstm_train_time

print(f"\n  Results:")
print(f"  - Accuracy:  {bilstm_metrics['accuracy']:.4f}")
print(f"  - Precision: {bilstm_metrics['precision']:.4f}")
print(f"  - Recall:    {bilstm_metrics['recall']:.4f}")
print(f"  - F1-Score:  {bilstm_metrics['f1_score']:.4f}")
print(f"  - AUC:       {bilstm_metrics['auc']:.4f}")

# Save model
model_bilstm.save('models/deep_learning/bilstm_final.keras')
print("  ✓ Model saved")

# ===== MODEL 3: LSTM WITH GLOBAL MAX POOLING =====
print("\n[6/7] Building and training LSTM with Global Max Pooling...")

print("\n  Architecture:")
print("  - Embedding Layer (trainable)")
print("  - LSTM (128 units, return_sequences=True)")
print("  - Global Max Pooling")
print("  - Dense Output (sigmoid)")

model_lstm_pool = Sequential([
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=True
    ),
    SpatialDropout1D(0.2),
    LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_lstm_pool.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
print("\n  Training...")
start_time = time.time()

history_lstm_pool = model_lstm_pool.fit(
    X_train_pad, y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

lstm_pool_train_time = time.time() - start_time
print(f"\n  ✓ Training completed in {lstm_pool_train_time:.2f} seconds")

# Evaluate
y_pred_proba_lstm_pool = model_lstm_pool.predict(X_test_pad, verbose=0).flatten()
y_pred_lstm_pool = (y_pred_proba_lstm_pool > 0.5).astype(int)

lstm_pool_metrics = evaluate_model(y_test, y_pred_lstm_pool, y_pred_proba_lstm_pool, "LSTM + MaxPooling")
lstm_pool_metrics['training_time'] = lstm_pool_train_time

print(f"\n  Results:")
print(f"  - Accuracy:  {lstm_pool_metrics['accuracy']:.4f}")
print(f"  - Precision: {lstm_pool_metrics['precision']:.4f}")
print(f"  - Recall:    {lstm_pool_metrics['recall']:.4f}")
print(f"  - F1-Score:  {lstm_pool_metrics['f1_score']:.4f}")
print(f"  - AUC:       {lstm_pool_metrics['auc']:.4f}")

# Save model
model_lstm_pool.save('models/deep_learning/lstm_pool_final.keras')
print("  ✓ Model saved")

# ===== COMPARISON =====
print("\n[7/7] Comparing deep learning models...")

all_dl_results = [lstm_metrics, bilstm_metrics, lstm_pool_metrics]

# Create comparison table
comparison_df = pd.DataFrame(all_dl_results)
comparison_df = comparison_df[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'auc', 'training_time']]

print("\n" + "="*90)
print("DEEP LEARNING MODELS COMPARISON")
print("="*90)
print(comparison_df.to_string(index=False))
print("="*90)

# Save comparison
comparison_df.to_csv('results/deep_learning/model_comparison.csv', index=False)

# Save detailed results
with open('results/deep_learning/detailed_results.json', 'w') as f:
    json.dump(all_dl_results, f, indent=2)

# Create comparison visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'training_time']
colors = ['#3498db', '#e74c3c', '#2ecc71']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    values = [m[metric] for m in all_dl_results]
    models = [m['model'] for m in all_dl_results]
    
    bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    
    if metric == 'training_time':
        ax.set_ylabel('Time (seconds)', fontsize=12)
    else:
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)
    
    # Add value labels
    for i, v in enumerate(values):
        if metric == 'training_time':
            ax.text(i, v + 5, f'{v:.1f}s', ha='center', fontweight='bold', fontsize=9)
        else:
            ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('results/deep_learning/models_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Comparison visualization saved")

# Compare all training histories
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

histories = [
    (history_lstm.history, 'LSTM'),
    (history_bilstm.history, 'Bidirectional LSTM'),
    (history_lstm_pool.history, 'LSTM + MaxPooling')
]

for idx, (hist, name) in enumerate(histories):
    axes[idx].plot(hist['accuracy'], label='Train Accuracy', linewidth=2)
    axes[idx].plot(hist['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[idx].set_title(f'{name} - Training History', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('Accuracy')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/deep_learning/all_training_histories.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ All training histories saved")

# Determine best model
best_model = max(all_dl_results, key=lambda x: x['f1_score'])

print("\n" + "="*60)
print(f"BEST DEEP LEARNING MODEL: {best_model['model']}")
print(f"  - F1-Score: {best_model['f1_score']:.4f}")
print(f"  - Accuracy: {best_model['accuracy']:.4f}")
print(f"  - AUC: {best_model['auc']:.4f}")
print("="*60)

print("\n✓ Deep learning model training complete!")
print("✓ All results saved in results/deep_learning/")
