# Sentiment Analysis of IMDB Movie Reviews
## Project Overview

This repository contains the **implementation, experimentation, and evaluation code** for a sentiment analysis system trained on the IMDB Movie Reviews dataset.
The project explores two key modeling paradigms:

1. **Traditional Machine Learning** — Logistic Regression using TF-IDF vectorization.
2. **Deep Learning** — Bidirectional LSTM (BiLSTM) network using Word2Vec embeddings.

The objective was to evaluate both approaches in terms of **performance, interpretability, and computational efficiency**, and determine the most appropriate model for text-based sentiment classification tasks.

## Repository Structure

```
Sentiment-Analysis/
│
├── data/                         # Dataset storage (loaded via Hugging Face)
│
├── notebooks/                    # Interactive notebooks for EDA and experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_traditional_ml.ipynb
│   └── 04_deep_learning.ipynb
│
├── scripts/                      # Modular Python scripts
│   ├── 00_utils.py               # Utility functions for loading/saving data
│   ├── 01_data_cleaning.py       # Preprocessing pipeline
│   ├── 02_feature_engineering.py # TF-IDF and Word2Vec embeddings
│   ├── 03_train_lr.py            # Logistic Regression training
│   ├── 04_train_lstm.py          # LSTM model training and tuning
│   ├── 05_evaluate_models.py     # Model evaluation and comparison
│   └── 06_visualizations.py      # Performance and data visualizations
│
├── models/                       # Serialized models (e.g., .pkl and .h5 files)
│   ├── logistic_regression_tfidf.pkl
│   └── bidirectional_lstm_word2vec.h5
│
├── requirements.txt              # Python dependencies
│
├── README.md                     # Project documentation (this file)
│
└── report/                       # Academic report and deliverables
    └── Sentiment_Analysis_Report.pdf
```

## Setup and Installation

### Prerequisites

Ensure you have **Python 3.9+** and **pip** installed.
It is strongly recommended to use a virtual environment.

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/JohnOngeri/Sentiment-Analysis.git
cd Sentiment-Analysis

# Create and activate a virtual environment (Mac/Linux)
python3 -m venv venv
source venv/bin/activate

# or for Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The project primarily uses:

* `pandas`, `numpy` — data handling
* `scikit-learn` — traditional ML models
* `tensorflow`, `keras` — deep learning (LSTM implementation)
* `gensim` — Word2Vec embeddings
* `matplotlib`, `seaborn` — visualization

## Data Preprocessing Pipeline

Preprocessing steps (implemented in `01_data_cleaning.py`):

1. HTML tag removal
2. Lowercasing and punctuation removal
3. Tokenization
4. Stopword elimination using NLTK
5. Lemmatization
6. Optional length filtering

Outputs are stored as cleaned token lists or joined sentences for later embedding generation.

## Feature Engineering

Two main feature extraction strategies were employed:

* **TF-IDF Vectorization** (for Logistic Regression):

  * Unigrams and bigrams considered
  * Minimum document frequency: 5
  * Maximum features: 50,000

* **Word2Vec Embeddings** (for BiLSTM):

  * 100-dimensional vectors trained on the IMDB corpus
  * Skip-gram architecture
  * Used to initialize embedding weights in the neural model

## Model Training and Evaluation

### 1. Logistic Regression

* Vectorizer: TF-IDF
* Regularization: L2
* Solver: `liblinear`
* Evaluation metrics: Accuracy, Precision, Recall, F1-Score

### 2. Bidirectional LSTM

* Embedding layer: 100D Word2Vec (trainable)
* Recurrent layer: BiLSTM (128 units)
* Dense: ReLU + Dropout (0.5)
* Output: Sigmoid
* Optimizer: Adam
* Loss: Binary Cross-Entropy
* Metrics: Accuracy, Precision, Recall, F1-Score, AUC

## Experimental Configuration

* **Hardware:** Apple M-series chip, 8-core CPU
* **Frameworks:** TensorFlow 2.x, Scikit-learn 1.5
* **Training Setup:**

  * Epochs: 5–10
  * Batch size: 64 / 128
  * Learning rates tested: 0.001, 0.0005
* **Hyperparameter Tuning:** Grid search for LR; manual tuning for BiLSTM.

## Performance Summary

| Model                         | Accuracy | Precision | Recall | F1-Score |
| :---------------------------- | :------: | :-------: | :----: | :------: |
| Logistic Regression (TF-IDF)  |   0.88   |    0.87   |  0.88  | **0.88** |
| Bidirectional LSTM (Word2Vec) |   0.86   |    0.90   |  0.81  |   0.85   |
| SVM (TF-IDF)                  |   0.86   |    0.87   |  0.86  |   0.86   |
| Naïve Bayes                   |   0.85   |    0.85   |  0.84  |   0.85   |

➡️ **Key insight:** Logistic Regression outperformed BiLSTM slightly while being far more computationally efficient.

## Reproducibility

To reproduce the experiments:

```bash
# Run traditional ML pipeline
python scripts/03_train_lr.py

# Run deep learning pipeline
python scripts/04_train_lstm.py

# Evaluate models
python scripts/05_evaluate_models.py
```

To reproduce EDA visualizations:

```bash
python scripts/06_visualizations.py
```

## Results and Artifacts

* Trained models are stored in `/models/`
* Visualizations and metrics are generated in `/notebooks/` and `/outputs/`
* Preprocessing logs and performance summaries are automatically saved in `/logs/`

## Team Contributions

| Member            | Role                           | Contributions                                           |
| :---------------- | :----------------------------- | :------------------------------------------------------ |
| **John Ouma**     | Data & EDA                     | Data exploration, visualization, and summary statistics |
| **Omar Keita**    | Preprocessing & Traditional ML | Text cleaning pipeline, TF-IDF modeling                 |
| **Lesly Ndizeye** | Deep Learning & Experiments    | Word2Vec embedding training, BiLSTM architecture        |
| **Ivan Shema**    | Evaluation & Documentation     | Model comparison, performance analysis, report writing  |


## References

* Maas et al. (2011). *Learning Word Vectors for Sentiment Analysis*.
* Mikolov et al. (2013). *Efficient Estimation of Word Representations in Vector Space.*
* Hugging Face Datasets: [IMDB Reviews](https://huggingface.co/datasets/imdb)

## Summary

This repository provides all the implementation assets for replicating and extending our sentiment analysis study.
While both modeling approaches achieved strong results, **TF-IDF + Logistic Regression** remains a practical and high-performing baseline for binary sentiment classification, especially where interpretability and efficiency are key requirements.
