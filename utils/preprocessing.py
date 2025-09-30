"""
Text preprocessing utilities for sentiment analysis
Includes cleaning, tokenization, and embedding generation
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline
    """
    
    def __init__(self, remove_stopwords=True, lemmatize=True, lowercase=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        
    def clean_text(self, text):
        """
        Clean text by removing HTML tags, special characters, and extra whitespace
        """
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters and digits (keep letters and basic punctuation)
        text = re.sub(r'[^a-zA-Z\s\.\!\?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words
        """
        return word_tokenize(text)
    
    def remove_stop_words(self, tokens):
        """
        Remove stopwords from token list
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """
        Lemmatize tokens to their base form
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        """
        # Clean text
        text = self.clean_text(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        
        # Lemmatize
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts):
        """
        Preprocess a batch of texts
        """
        return [self.preprocess(text) for text in texts]


def load_glove_embeddings(glove_path, embedding_dim=100):
    """
    Load pre-trained GloVe embeddings
    
    Args:
        glove_path: Path to GloVe file (e.g., 'glove.6B.100d.txt')
        embedding_dim: Dimension of embeddings (50, 100, 200, or 300)
    
    Returns:
        Dictionary mapping words to embedding vectors
    """
    embeddings_index = {}
    
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = vector
    except FileNotFoundError:
        print(f"Warning: GloVe file not found at {glove_path}")
        print("Please download GloVe embeddings from: https://nlp.stanford.edu/projects/glove/")
        return None
    
    print(f"Loaded {len(embeddings_index)} word vectors from GloVe")
    return embeddings_index


def create_embedding_matrix(word_index, embeddings_index, embedding_dim=100, max_words=10000):
    """
    Create embedding matrix for neural network
    
    Args:
        word_index: Dictionary mapping words to indices
        embeddings_index: Dictionary of pre-trained embeddings
        embedding_dim: Dimension of embeddings
        max_words: Maximum number of words to include
    
    Returns:
        Embedding matrix of shape (max_words, embedding_dim)
    """
    embedding_matrix = np.zeros((max_words, embedding_dim))
    
    for word, i in word_index.items():
        if i >= max_words:
            continue
        
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix
