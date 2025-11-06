import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json
from datetime import datetime
from model_architecture import create_cnn_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

class CNNImageFraudTrainer:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.img_height = self.config.get('img_height', 224)
        self.img_width = self.config.get('img_width', 224)
        self.batch_size = self.config.get('batch_size', 32)
        self.epochs = self.config.get('epochs', 50)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.model_save_path = self.config.get('model_save_path', 'models/')
        
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self, data_dir):
        """Load and preprocess image data"""
        print("Loading and preprocessing data...")
        
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training'
        )
        
        validation_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation'
        )
        
        return train_generator, validation_generator
    
    def build_model(self):
        """Build CNN model architecture"""
        print("Building CNN model...")
        self.model = create_cnn_model(
            input_shape=(self.img_height, self.img_width, 3),
            num_classes=1
        )
        
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        print(self.model.summary())
        
    def setup_callbacks(self):
        """Setup training callbacks"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=os.path.join(self.model_save_path, f'cnn_fraud_best_{timestamp}.h5'),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=os.path.join('logs', timestamp),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        
        csv_logger = callbacks.CSVLogger(
            os.path.join('logs', f'training_log_{timestamp}.csv')
        )
        
        return [checkpoint_callback, early_stopping, reduce_lr, tensorboard_callback, csv_logger]
    
    def train(self, train_generator, validation_generator):
        """Train the model"""
        print("Starting training...")
        
        callback_list = self.setup_callbacks()
        
        self.history = self.model.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=validation_generator,
            callbacks=callback_list,
            verbose=1
        )
        
        print("Training completed!")
        
    def evaluate_model(self, test_generator):
        """Evaluate model performance"""
        print("Evaluating model...")
        
        test_loss, test_accuracy, test_auc, test_precision, test_recall = self.model.evaluate(
            test_generator,
            verbose=1
        )
        
        print(f"\nTest Results:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"AUC: {test_auc:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        
        # Generate predictions
        predictions = self.model.predict(test_generator)
        predicted_classes = (predictions > 0.5).astype(int)
        
        # Get true labels
        true_labels = test_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predicted_classes, 
                                   target_names=['Legitimate', 'Fraud']))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_classes)
        self.plot_confusion_matrix(cm, ['Legitimate', 'Fraud'])
        
        # ROC AUC
        roc_auc = roc_auc_score(true_labels, predictions)
        print(f"\nROC AUC Score: {roc_auc:.4f}")
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'roc_auc': roc_auc
        }
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC
        axes[1, 0].plot(self.history.history['auc'], label='Train AUC')
        axes[1, 0].plot(self.history.history['val_auc'], label='Val AUC')
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Precision & Recall
        axes[1, 1].plot(self.history.history['precision'], label='Train Precision')
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
        axes[1, 1].plot(self.history.history['val_precision'], label='Val Precision')
        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self, filename='cnn_fraud_final.h5'):
        """Save the trained model"""
        save_path = os.path.join(self.model_save_path, filename)
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
        
        # Save model architecture as JSON
        model_json = self.model.to_json()
        json_path = save_path.replace('.h5', '.json')
        with open(json_path, 'w') as json_file:
            json_file.write(model_json)
        print(f"Model architecture saved to {json_path}")

def main():
    # Configuration
    data_dir = 'data/fraud_images'
    config_path = 'config.json'
    
    # Initialize trainer
    trainer = CNNImageFraudTrainer(config_path)
    
    # Load data
    train_gen, val_gen = trainer.load_and_preprocess_data(data_dir)
    
    # Build model
    trainer.build_model()
    
    # Train model
    trainer.train(train_gen, val_gen)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate on test set
    test_gen = val_gen  # In practice, use a separate test set
    results = trainer.evaluate_model(test_gen)
    
    # Save model
    trainer.save_model()
    
    # Save results
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nTraining pipeline completed successfully!")

if __name__ == '__main__':
    main()
