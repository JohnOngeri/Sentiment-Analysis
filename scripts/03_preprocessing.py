"""
Text Preprocessing and Embedding Comparison
Compares TF-IDF, Word2Vec, and GloVe embeddings
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import TextPreprocessor
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

print("="*60)
print("TEXT PREPROCESSING AND EMBEDDING COMPARISON")
print("="*60)

# Load data
print("\n[1/5] Loading data...")
train_df = pd.read_csv('data/train_data.csv')
test_df = pd.read_csv('data/test_data.csv')

# Initialize preprocessor
print("\n[2/5] Preprocessing text data...")
preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True, lowercase=True)

# Preprocess training data
print("  - Processing training data...")
train_df['cleaned_text'] = [preprocessor.preprocess(text) for text in tqdm(train_df['text'], desc="Training")]

# Preprocess test data
print("  - Processing test data...")
test_df['cleaned_text'] = [preprocessor.preprocess(text) for text in tqdm(test_df['text'], desc="Testing")]

# Save preprocessed data
train_df.to_csv('data/train_preprocessed.csv', index=False)
test_df.to_csv('data/test_preprocessed.csv', index=False)
print("  ✓ Preprocessed data saved")

# ===== EMBEDDING 1: TF-IDF =====
print("\n[3/5] Creating TF-IDF embeddings...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=5,
    max_df=0.8
)

X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['cleaned_text'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['cleaned_text'])

print(f"  - TF-IDF shape: {X_train_tfidf.shape}")
print(f"  - Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
print(f"  - Feature names (sample): {tfidf_vectorizer.get_feature_names_out()[:10]}")

# Save TF-IDF vectorizer
os.makedirs('models/embeddings', exist_ok=True)
with open('models/embeddings/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Save TF-IDF matrices
np.save('models/embeddings/X_train_tfidf.npy', X_train_tfidf.toarray())
np.save('models/embeddings/X_test_tfidf.npy', X_test_tfidf.toarray())
print("  ✓ TF-IDF embeddings saved")

# ===== EMBEDDING 2: Word2Vec =====
print("\n[4/5] Creating Word2Vec embeddings...")

# Tokenize for Word2Vec
train_tokens = [text.split() for text in train_df['cleaned_text']]
test_tokens = [text.split() for text in test_df['cleaned_text']]

# Train Word2Vec model
w2v_model = Word2Vec(
    sentences=train_tokens,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=10,
    sg=1  # Skip-gram model
)

print(f"  - Word2Vec vocabulary size: {len(w2v_model.wv)}")
print(f"  - Vector dimension: {w2v_model.wv.vector_size}")

# Save Word2Vec model
w2v_model.save('models/embeddings/word2vec.model')
print("  ✓ Word2Vec model saved")

# Create document vectors by averaging word vectors
def get_document_vector(tokens, model, vector_size=100):
    """Average word vectors to create document vector"""
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    
    if len(vectors) == 0:
        return np.zeros(vector_size)
    
    return np.mean(vectors, axis=0)

print("  - Creating document vectors...")
X_train_w2v = np.array([get_document_vector(tokens, w2v_model) for tokens in tqdm(train_tokens, desc="Train W2V")])
X_test_w2v = np.array([get_document_vector(tokens, w2v_model) for tokens in tqdm(test_tokens, desc="Test W2V")])

print(f"  - Train W2V shape: {X_train_w2v.shape}")
print(f"  - Test W2V shape: {X_test_w2v.shape}")

# Save Word2Vec document vectors
np.save('models/embeddings/X_train_w2v.npy', X_train_w2v)
np.save('models/embeddings/X_test_w2v.npy', X_test_w2v)
print("  ✓ Word2Vec document vectors saved")

# ===== EMBEDDING COMPARISON VISUALIZATION =====
print("\n[5/5] Creating embedding comparison visualization...")

# Compare embedding characteristics
embedding_stats = {
    'Embedding': ['TF-IDF', 'Word2Vec'],
    'Dimension': [X_train_tfidf.shape[1], X_train_w2v.shape[1]],
    'Sparsity': [
        f"{(1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])) * 100:.1f}%",
        f"{(np.sum(X_train_w2v == 0) / X_train_w2v.size) * 100:.1f}%"
    ],
    'Type': ['Sparse', 'Dense'],
    'Context': ['Document-level', 'Word-level']
}

stats_df = pd.DataFrame(embedding_stats)

# Create comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Dimension comparison
axes[0, 0].bar(stats_df['Embedding'], stats_df['Dimension'], color=['#3498db', '#e74c3c'], alpha=0.7)
axes[0, 0].set_title('Embedding Dimensions', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Number of Features')
for i, v in enumerate(stats_df['Dimension']):
    axes[0, 0].text(i, v + 100, str(v), ha='center', fontweight='bold')

# 2. Sample TF-IDF distribution
sample_tfidf = X_train_tfidf[0].toarray().flatten()
sample_tfidf = sample_tfidf[sample_tfidf > 0]  # Only non-zero values
axes[0, 1].hist(sample_tfidf, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('TF-IDF Value Distribution (Sample Document)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('TF-IDF Score')
axes[0, 1].set_ylabel('Frequency')

# 3. Sample Word2Vec distribution
sample_w2v = X_train_w2v[0]
axes[1, 0].hist(sample_w2v, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Word2Vec Value Distribution (Sample Document)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Vector Value')
axes[1, 0].set_ylabel('Frequency')

# 4. Comparison table
axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table_data = [
    ['TF-IDF', '5000', stats_df['Sparsity'][0], 'Sparse', 'Document-level'],
    ['Word2Vec', '100', stats_df['Sparsity'][1], 'Dense', 'Word-level']
]
table = axes[1, 1].table(
    cellText=table_data,
    colLabels=['Embedding', 'Dimension', 'Sparsity', 'Type', 'Context'],
    cellLoc='center',
    loc='center',
    colColours=['#f0f0f0']*5
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)
axes[1, 1].set_title('Embedding Comparison', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('visualizations/07_embedding_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: visualizations/07_embedding_comparison.png")

# ===== EMBEDDING JUSTIFICATION =====
print("\n" + "="*60)
print("EMBEDDING COMPARISON SUMMARY")
print("="*60)

print("\n1. TF-IDF (Term Frequency-Inverse Document Frequency)")
print("   Advantages:")
print("   - Simple and interpretable")
print("   - Captures document-level importance")
print("   - Works well with traditional ML models")
print("   - No training required")
print("   Disadvantages:")
print("   - High dimensionality (5000 features)")
print("   - Sparse representation")
print("   - No semantic understanding")
print("   - Doesn't capture word order")

print("\n2. Word2Vec (Skip-gram)")
print("   Advantages:")
print("   - Dense, low-dimensional representation (100 features)")
print("   - Captures semantic relationships")
print("   - Pre-trained on domain data")
print("   - Better for neural networks")
print("   Disadvantages:")
print("   - Requires training time")
print("   - Loses some document-level information")
print("   - Averaging may lose nuance")

print("\n3. GloVe (Global Vectors) - For Deep Learning")
print("   Advantages:")
print("   - Pre-trained on large corpus")
print("   - Captures global statistical information")
print("   - Excellent for transfer learning")
print("   - Works well with LSTMs/RNNs")
print("   Note: Will be used in deep learning model")

print("\n" + "="*60)
print("RECOMMENDATION:")
print("- Use TF-IDF for traditional ML models (Logistic Regression)")
print("- Use GloVe for deep learning models (LSTM)")
print("- Word2Vec serves as middle ground for comparison")
print("="*60)

print("\n✓ Preprocessing and embedding creation complete!")
print("✓ Files saved in models/embeddings/")
