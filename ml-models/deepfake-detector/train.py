import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from model_architecture import DeepfakeDetectorModel
from preprocessing import VideoPreprocessor
import os
import json
from datetime import datetime

class DeepfakeTrainer:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model = None
        self.preprocessor = VideoPreprocessor()
        self.history = None
        
    def build_model(self):
        """Build the deepfake detection model"""
        self.model = DeepfakeDetectorModel(
            input_shape=self.config['input_shape'],
            num_classes=self.config['num_classes']
        ).build()
        
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config['learning_rate']
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        
        return self.model
    
    def create_data_generators(self):
        """Create training and validation data generators"""
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )
        
        train_generator = train_datagen.flow_from_directory(
            self.config['train_data_path'],
            target_size=self.config['input_shape'][:2],
            batch_size=self.config['batch_size'],
            class_mode='binary'
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.config['val_data_path'],
            target_size=self.config['input_shape'][:2],
            batch_size=self.config['batch_size'],
            class_mode='binary'
        )
        
        return train_generator, val_generator
    
    def get_callbacks(self):
        """Define training callbacks"""
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config['checkpoint_dir'], 
                                     'model_{epoch:02d}_{val_accuracy:.4f}.h5'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.config['log_dir'], 
                                    datetime.now().strftime('%Y%m%d-%H%M%S')),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            ),
            keras.callbacks.CSVLogger(
                os.path.join(self.config['log_dir'], 'training_log.csv'),
                append=True
            )
        ]
        
        return callbacks
    
    def train(self):
        """Train the deepfake detection model"""
        print("Building model...")
        self.build_model()
        
        print("Creating data generators...")
        train_gen, val_gen = self.create_data_generators()
        
        print("Starting training...")
        self.history = self.model.fit(
            train_gen,
            epochs=self.config['epochs'],
            validation_data=val_gen,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate(self, test_data_path):
        """Evaluate model on test data"""
        test_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_data_path,
            target_size=self.config['input_shape'][:2],
            batch_size=self.config['batch_size'],
            class_mode='binary',
            shuffle=False
        )
        
        results = self.model.evaluate(test_generator, verbose=1)
        
        metrics = dict(zip(self.model.metrics_names, results))
        print("\nTest Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """Save trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load pre-trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

if __name__ == '__main__':
    trainer = DeepfakeTrainer()
    trainer.train()
    trainer.evaluate(trainer.config['test_data_path'])
    trainer.save_model(os.path.join(trainer.config['model_dir'], 'final_model.h5'))
