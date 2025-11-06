import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuralNetworkEnsemble:
    def __init__(self, model_configs=None):
        self.models = []
        self.scaler = StandardScaler()
        self.ensemble_weights = []
        self.model_configs = model_configs or self.get_default_configs()
        
    def get_default_configs(self):
        return [
            {'name': 'deep_nn', 'layers': [256, 128, 64, 32], 'dropout': 0.3},
            {'name': 'wide_nn', 'layers': [512, 256], 'dropout': 0.2},
            {'name': 'residual_nn', 'layers': [128, 128, 128], 'dropout': 0.25}
        ]
    
    def build_deep_neural_network(self, input_dim, config):
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.BatchNormalization()
        ])
        
        for units in config['layers']:
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.Dropout(config['dropout']))
            model.add(keras.layers.BatchNormalization())
        
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        return model
    
    def build_residual_network(self, input_dim, config):
        inputs = keras.Input(shape=(input_dim,))
        x = keras.layers.BatchNormalization()(inputs)
        
        for i, units in enumerate(config['layers']):
            residual = x
            x = keras.layers.Dense(units, activation='relu')(x)
            x = keras.layers.Dropout(config['dropout'])(x)
            x = keras.layers.BatchNormalization()(x)
            
            if i > 0 and x.shape[-1] == residual.shape[-1]:
                x = keras.layers.Add()([x, residual])
        
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model
