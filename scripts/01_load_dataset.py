"""
Script to load and explore the IMDB movie reviews dataset
Dataset: IMDB Movie Reviews (50,000 reviews)
Source: Stanford AI Lab - http://ai.stanford.edu/~amaas/data/sentiment/
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import json

print("Loading IMDB dataset...")

# Load IMDB dataset from Hugging Face
dataset = load_dataset("imdb")

# Convert to pandas DataFrames
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"\n=== Dataset Overview ===")
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Total samples: {len(train_df) + len(test_df)}")

print(f"\n=== Dataset Structure ===")
print(train_df.head())

print(f"\n=== Label Distribution (Training) ===")
print(train_df['label'].value_counts())
print(f"\nClass Balance: {train_df['label'].value_counts(normalize=True)}")

print(f"\n=== Sample Reviews ===")
print(f"\nPositive Review Example:")
print(train_df[train_df['label'] == 1]['text'].iloc[0][:300] + "...")

print(f"\nNegative Review Example:")
print(train_df[train_df['label'] == 0]['text'].iloc[0][:300] + "...")

# Calculate review lengths
train_df['review_length'] = train_df['text'].apply(lambda x: len(x.split()))
test_df['review_length'] = test_df['text'].apply(lambda x: len(x.split()))

print(f"\n=== Review Length Statistics ===")
print(train_df['review_length'].describe())

# Save processed data
train_df.to_csv('data/train_data.csv', index=False)
test_df.to_csv('data/test_data.csv', index=False)

# Save summary statistics
stats = {
    'dataset_name': 'IMDB Movie Reviews',
    'source': 'Stanford AI Lab',
    'total_samples': len(train_df) + len(test_df),
    'train_samples': len(train_df),
    'test_samples': len(test_df),
    'num_classes': 2,
    'classes': ['Negative', 'Positive'],
    'class_balance': train_df['label'].value_counts(normalize=True).to_dict(),
    'avg_review_length': float(train_df['review_length'].mean()),
    'median_review_length': float(train_df['review_length'].median())
}

with open('data/dataset_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\n✓ Dataset loaded and saved successfully!")
print("✓ Files saved: data/train_data.csv, data/test_data.csv, data/dataset_stats.json")
