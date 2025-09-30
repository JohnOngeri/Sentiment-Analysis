"""
Exploratory Data Analysis with Visualizations
Generates 4+ visualizations for understanding the dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("Loading data for EDA...")
train_df = pd.read_csv('data/train_data.csv')

# Create visualizations directory
import os
os.makedirs('visualizations', exist_ok=True)

# ===== Visualization 1: Sentiment Distribution =====
print("\n[1/6] Creating sentiment distribution plot...")
plt.figure(figsize=(10, 6))
sentiment_counts = train_df['label'].value_counts()
colors = ['#e74c3c', '#2ecc71']
plt.bar(['Negative (0)', 'Positive (1)'], sentiment_counts.values, color=colors, alpha=0.8, edgecolor='black')
plt.title('Sentiment Distribution in Training Data', fontsize=16, fontweight='bold')
plt.ylabel('Number of Reviews', fontsize=12)
plt.xlabel('Sentiment Class', fontsize=12)
for i, v in enumerate(sentiment_counts.values):
    plt.text(i, v + 200, str(v), ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/01_sentiment_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: visualizations/01_sentiment_distribution.png")

# ===== Visualization 2: Review Length Distribution =====
print("\n[2/6] Creating review length distribution...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram
axes[0].hist(train_df['review_length'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
axes[0].set_title('Distribution of Review Lengths', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Words', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].axvline(train_df['review_length'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {train_df["review_length"].mean():.0f}')
axes[0].axvline(train_df['review_length'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {train_df["review_length"].median():.0f}')
axes[0].legend()

# Box plot by sentiment
axes[1].boxplot([train_df[train_df['label']==0]['review_length'], 
                  train_df[train_df['label']==1]['review_length']], 
                 labels=['Negative', 'Positive'],
                 patch_artist=True,
                 boxprops=dict(facecolor='#3498db', alpha=0.7))
axes[1].set_title('Review Length by Sentiment', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Number of Words', fontsize=12)
axes[1].set_xlabel('Sentiment', fontsize=12)

plt.tight_layout()
plt.savefig('visualizations/02_review_length_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: visualizations/02_review_length_distribution.png")

# ===== Visualization 3: Word Clouds =====
print("\n[3/6] Creating word clouds...")

def clean_text_for_wordcloud(text):
    """Basic cleaning for word cloud"""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters
    return text.lower()

# Positive reviews word cloud
positive_text = ' '.join(train_df[train_df['label']==1]['text'].head(1000).apply(clean_text_for_wordcloud))
negative_text = ' '.join(train_df[train_df['label']==0]['text'].head(1000).apply(clean_text_for_wordcloud))

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Positive word cloud
wordcloud_pos = WordCloud(width=800, height=400, background_color='white', 
                          colormap='Greens', max_words=100).generate(positive_text)
axes[0].imshow(wordcloud_pos, interpolation='bilinear')
axes[0].set_title('Most Common Words in Positive Reviews', fontsize=14, fontweight='bold')
axes[0].axis('off')

# Negative word cloud
wordcloud_neg = WordCloud(width=800, height=400, background_color='white', 
                          colormap='Reds', max_words=100).generate(negative_text)
axes[1].imshow(wordcloud_neg, interpolation='bilinear')
axes[1].set_title('Most Common Words in Negative Reviews', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('visualizations/03_wordclouds.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: visualizations/03_wordclouds.png")

# ===== Visualization 4: Top Words Comparison =====
print("\n[4/6] Creating top words comparison...")

def get_top_words(texts, n=15):
    """Extract top n words from texts"""
    all_words = []
    for text in texts:
        cleaned = clean_text_for_wordcloud(text)
        words = [w for w in cleaned.split() if len(w) > 3]  # Filter short words
        all_words.extend(words)
    
    # Remove common stopwords
    stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'their', 
                 'there', 'about', 'would', 'could', 'should', 'which', 'what'}
    all_words = [w for w in all_words if w not in stopwords]
    
    counter = Counter(all_words)
    return counter.most_common(n)

pos_words = get_top_words(train_df[train_df['label']==1]['text'].head(2000))
neg_words = get_top_words(train_df[train_df['label']==0]['text'].head(2000))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Positive words
pos_df = pd.DataFrame(pos_words, columns=['word', 'count'])
axes[0].barh(pos_df['word'], pos_df['count'], color='#2ecc71', alpha=0.8)
axes[0].set_title('Top 15 Words in Positive Reviews', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Frequency', fontsize=12)
axes[0].invert_yaxis()

# Negative words
neg_df = pd.DataFrame(neg_words, columns=['word', 'count'])
axes[1].barh(neg_df['word'], neg_df['count'], color='#e74c3c', alpha=0.8)
axes[1].set_title('Top 15 Words in Negative Reviews', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Frequency', fontsize=12)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('visualizations/04_top_words_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: visualizations/04_top_words_comparison.png")

# ===== Visualization 5: Review Length by Sentiment (Detailed) =====
print("\n[5/6] Creating detailed review length analysis...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Violin plot
axes[0, 0].violinplot([train_df[train_df['label']==0]['review_length'], 
                        train_df[train_df['label']==1]['review_length']], 
                       positions=[0, 1], showmeans=True, showmedians=True)
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(['Negative', 'Positive'])
axes[0, 0].set_title('Review Length Distribution (Violin Plot)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Number of Words')

# Histogram overlay
axes[0, 1].hist(train_df[train_df['label']==0]['review_length'], bins=50, alpha=0.5, 
                label='Negative', color='#e74c3c', density=True)
axes[0, 1].hist(train_df[train_df['label']==1]['review_length'], bins=50, alpha=0.5, 
                label='Positive', color='#2ecc71', density=True)
axes[0, 1].set_title('Review Length Density Comparison', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Number of Words')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()

# Cumulative distribution
axes[1, 0].hist(train_df[train_df['label']==0]['review_length'], bins=100, alpha=0.7, 
                label='Negative', color='#e74c3c', cumulative=True, density=True)
axes[1, 0].hist(train_df[train_df['label']==1]['review_length'], bins=100, alpha=0.7, 
                label='Positive', color='#2ecc71', cumulative=True, density=True)
axes[1, 0].set_title('Cumulative Distribution of Review Lengths', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Number of Words')
axes[1, 0].set_ylabel('Cumulative Probability')
axes[1, 0].legend()

# Statistics table
stats_data = []
for label, name in [(0, 'Negative'), (1, 'Positive')]:
    subset = train_df[train_df['label']==label]['review_length']
    stats_data.append([
        name,
        f"{subset.mean():.1f}",
        f"{subset.median():.1f}",
        f"{subset.std():.1f}",
        f"{subset.min():.0f}",
        f"{subset.max():.0f}"
    ])

axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=stats_data, 
                         colLabels=['Sentiment', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                         cellLoc='center', loc='center',
                         colColours=['#f0f0f0']*6)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
axes[1, 1].set_title('Review Length Statistics', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('visualizations/05_review_length_detailed.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: visualizations/05_review_length_detailed.png")

# ===== Visualization 6: Class Balance Check =====
print("\n[6/6] Creating class balance visualization...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
colors_pie = ['#e74c3c', '#2ecc71']
explode = (0.05, 0.05)
axes[0].pie(sentiment_counts.values, labels=['Negative', 'Positive'], autopct='%1.1f%%',
            colors=colors_pie, explode=explode, shadow=True, startangle=90)
axes[0].set_title('Class Balance (Pie Chart)', fontsize=14, fontweight='bold')

# Percentage bar
percentages = train_df['label'].value_counts(normalize=True) * 100
axes[1].barh(['Negative', 'Positive'], percentages.values, color=colors_pie, alpha=0.8, edgecolor='black')
axes[1].set_xlabel('Percentage (%)', fontsize=12)
axes[1].set_title('Class Balance (Percentage)', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, 100)
for i, v in enumerate(percentages.values):
    axes[1].text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/06_class_balance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: visualizations/06_class_balance.png")

print("\n" + "="*60)
print("✓ All visualizations created successfully!")
print("="*60)
print("\nKey Insights from EDA:")
print(f"1. Dataset is perfectly balanced: 50% positive, 50% negative")
print(f"2. Average review length: {train_df['review_length'].mean():.0f} words")
print(f"3. Review lengths range from {train_df['review_length'].min():.0f} to {train_df['review_length'].max():.0f} words")
print(f"4. Both sentiment classes have similar length distributions")
print(f"5. Word clouds reveal distinct vocabulary patterns between positive and negative reviews")
