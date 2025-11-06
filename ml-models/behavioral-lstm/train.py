import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

class BehavioralLSTMTrainer:
    def __init__(self, sequence_length=50, feature_dim=20):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = None
        
    def build_model(self):
        inputs = keras.Input(shape=(self.sequence_length, self.feature_dim))
        
        x = layers.LSTM(128, return_sequences=True)(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(32)(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
    def prepare_sequences(self, data):
        sequences = []
        labels = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            label = data[i + self.sequence_length]['is_fraud']
            sequences.append(seq)
            labels.append(label)
            
        return np.array(sequences), np.array(labels)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'auc': results[2]
        }
        return metrics
    
    def save_model(self, path):
        self.model.save(path)
        
    def load_model(self, path):
        self.model = keras.models.load_model(path)

if __name__ == '__main__':
    trainer = BehavioralLSTMTrainer()
    trainer.build_model()
    print(trainer.model.summary())
