# Sentiment Analysis on IMDB Movie Reviews

A comprehensive sentiment analysis project implementing both traditional machine learning and deep learning approaches to classify movie reviews as positive or negative.

## Project Overview

This project analyzes the IMDB Movie Reviews dataset (50,000 reviews) using multiple machine learning techniques, comparing traditional ML models with deep learning architectures.

**Team Members:** [Add your names here]  
**Course:** [Add course name]  
**Date:** [Add date]

## Dataset

- **Source:** IMDB Movie Reviews Dataset (Stanford AI Lab)
- **Size:** 50,000 reviews (25,000 training, 25,000 testing)
- **Classes:** Binary (Positive/Negative)
- **Balance:** Perfectly balanced (50% each class)

## Project Structure

\`\`\`
sentiment-analysis-project/
├── data/                          # Dataset files
│   ├── train_data.csv
│   ├── test_data.csv
│   └── dataset_stats.json
├── models/                        # Saved models
│   ├── traditional_ml/
│   └── deep_learning/
├── visualizations/                # EDA plots and results
│   ├── 01_sentiment_distribution.png
│   ├── 02_review_length_distribution.png
│   ├── 03_wordclouds.png
│   ├── 04_top_words_comparison.png
│   ├── 05_review_length_detailed.png
│   └── 06_class_balance.png
├── results/                       # Experiment results
│   ├── experiment_tables/
│   └── evaluation_metrics/
├── scripts/                       # Python scripts
│   ├── 01_load_dataset.py
│   ├── 02_eda_visualizations.py
│   ├── 03_preprocessing.py
│   ├── 04_traditional_ml.py
│   ├── 05_deep_learning.py
│   └── 06_evaluation.py
├── notebooks/                     # Jupyter notebooks
│   └── sentiment_analysis.ipynb
├── utils/                         # Utility functions
│   ├── preprocessing.py
│   └── evaluation.py
├── report/                        # Final report
│   └── sentiment_analysis_report.pdf
└── README.md
\`\`\`

## Setup Instructions

### Prerequisites

\`\`\`bash
Python 3.8+
pip or conda package manager
\`\`\`

### Installation

1. Clone the repository:
\`\`\`bash
git clone <repository-url>
cd sentiment-analysis-project
\`\`\`

2. Install required packages:
\`\`\`bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install tensorflow keras torch torchvision
pip install datasets transformers wordcloud
pip install gensim nltk
\`\`\`

3. Download NLTK data:
\`\`\`python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
\`\`\`

## Running the Project

Execute scripts in order:

\`\`\`bash
# 1. Load and explore dataset
python scripts/01_load_dataset.py

# 2. Generate EDA visualizations
python scripts/02_eda_visualizations.py

# 3. Preprocess data and create embeddings
python scripts/03_preprocessing.py

# 4. Train traditional ML models
python scripts/04_traditional_ml.py

# 5. Train deep learning models
python scripts/05_deep_learning.py

# 6. Evaluate and compare models
python scripts/06_evaluation.py
\`\`\`

## Models Implemented

### Traditional Machine Learning
- **Logistic Regression** with TF-IDF features
- Hyperparameters: C=1.0, max_iter=1000, solver='lbfgs'

### Deep Learning
- **LSTM (Long Short-Term Memory)** with GloVe embeddings
- Architecture: Embedding → LSTM(128) → Dropout(0.5) → Dense(1)
- Optimizer: Adam, Loss: Binary Crossentropy

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | TBD | TBD | TBD | TBD |
| LSTM | TBD | TBD | TBD | TBD |

*Results will be populated after running experiments*

## Key Findings

1. **Dataset Balance:** Perfectly balanced dataset eliminates class imbalance concerns
2. **Review Length:** Average review length is ~230 words with high variance
3. **Vocabulary Patterns:** Distinct word usage between positive and negative reviews
4. **Model Performance:** [To be filled after experiments]

## Experiment Tables

### Experiment 1: Learning Rate Comparison
| Learning Rate | Batch Size | Optimizer | Accuracy | F1-Score | Loss |
|---------------|------------|-----------|----------|----------|------|
| TBD | TBD | TBD | TBD | TBD | TBD |

### Experiment 2: Embedding Comparison
| Embedding | Model | Accuracy | F1-Score | Training Time |
|-----------|-------|----------|----------|---------------|
| TBD | TBD | TBD | TBD | TBD |

## Technologies Used

- **Languages:** Python 3.8+
- **ML Libraries:** scikit-learn, TensorFlow/Keras, PyTorch
- **Data Processing:** pandas, numpy, NLTK
- **Visualization:** matplotlib, seaborn, wordcloud
- **Embeddings:** TF-IDF, Word2Vec, GloVe

## References

1. Maas, A. L., et al. (2011). "Learning Word Vectors for Sentiment Analysis." ACL.
2. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation.
3. Pennington, J., et al. (2014). "GloVe: Global Vectors for Word Representation." EMNLP.

## License

This project is for educational purposes only.

## Contact

[Add your contact information]
