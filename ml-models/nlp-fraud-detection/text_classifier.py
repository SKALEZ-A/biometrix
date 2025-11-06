import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class TextFraudClassifier:
    def __init__(self, max_words: int = 10000, max_len: int = 100):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words)
        self.model = None
        
    def build_model(self, embedding_dim: int = 128):
        """Build LSTM-based text classification model"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.max_words, embedding_dim, input_length=self.max_len),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info("Text fraud classifier model built")
        
    def preprocess_texts(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """Preprocess text data"""
        if fit:
            self.tokenizer.fit_on_texts(texts)
            
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        return padded
        
    def train(self, texts: List[str], labels: np.ndarray, 
             validation_split: float = 0.2, epochs: int = 10, batch_size: int = 32):
        """Train the text classifier"""
        X = self.preprocess_texts(texts, fit=True)
        
        history = self.model.fit(
            X, labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        logger.info("Training completed")
        return history
        
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict fraud probability for texts"""
        X = self.preprocess_texts(texts, fit=False)
        predictions = self.model.predict(X, verbose=0)
        return predictions
        
    def extract_fraud_indicators(self, text: str) -> List[str]:
        """Extract fraud indicators from text"""
        fraud_keywords = [
            'urgent', 'verify', 'suspended', 'confirm', 'click here',
            'limited time', 'act now', 'congratulations', 'winner',
            'free', 'prize', 'claim', 'password', 'account', 'security'
        ]
        
        text_lower = text.lower()
        found_indicators = [keyword for keyword in fraud_keywords if keyword in text_lower]
        
        return found_indicators
        
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (simplified)"""
        positive_words = ['thank', 'great', 'excellent', 'good', 'happy']
        negative_words = ['urgent', 'problem', 'issue', 'suspended', 'error']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.5
            
        return positive_count / (positive_count + negative_count)
