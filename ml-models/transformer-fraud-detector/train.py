import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
from tqdm import tqdm

from model import (
    create_fraud_detector,
    FraudDetectionLoss,
    TransactionSequenceEncoder
)


class FraudDataset(Dataset):
    """Dataset for fraud detection training"""
    
    def __init__(
        self,
        data_path: str,
        seq_length: int = 50,
        transform=None
    ):
        self.data = self._load_data(data_path)
        self.seq_length = seq_length
        self.transform = transform
    
    def _load_data(self, path: str) -> List[Dict]:
        """Load and preprocess data"""
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        
        # Extract features and label
        features = torch.tensor(sample['features'], dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        # Pad or truncate to seq_length
        if features.shape[0] < self.seq_length:
            padding = torch.zeros(self.seq_length - features.shape[0], features.shape[1])
            features = torch.cat([features, padding], dim=0)
        else:
            features = features[:self.seq_length]
        
        if self.transform:
            features = self.transform(features)
        
        return features, label


class FraudDetectorTrainer:
    """Trainer class for fraud detection models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        class_weights: Optional[torch.Tensor] = None,
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = FraudDetectionLoss(
            alpha=0.25,
            gamma=2.0,
            class_weights=class_weights.to(device) if class_weights is not None else None
        )
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Tracking
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.epoch = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch} [Train]')
        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(features)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = self._calculate_metrics(all_labels, all_preds)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.epoch} [Val]')
            for features, labels in pbar:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, _ = self.model(features)
                loss = self.criterion(logits, labels)
                
                # Track metrics
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _calculate_metrics(
        self,
        labels: List[int],
        preds: List[int],
        probs: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
        }
        
        if probs is not None:
            metrics['auc_roc'] = roc_auc_score(labels, probs)
        
        return metrics
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10):
        """Train the model for multiple epochs"""
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        self.writer.close()
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to tensorboard"""
        for name, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{name}', value, self.epoch)
        
        for name, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{name}', value, self.epoch)
        
        print(f"\nEpoch {self.epoch}:")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, "
              f"Precision: {train_metrics['precision']:.4f}, "
              f"Recall: {train_metrics['recall']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}")
        if 'auc_roc' in val_metrics:
            print(f"Val AUC-ROC: {val_metrics['auc_roc']:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
        }
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_f1 = checkpoint['best_val_f1']
        
        print(f"Checkpoint loaded: {path}")


def main():
    """Main training function"""
    # Configuration
    config = {
        'input_dim': 128,
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 4,
        'batch_size': 64,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'seq_length': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create datasets
    train_dataset = FraudDataset(
        data_path='data/train.json',
        seq_length=config['seq_length']
    )
    val_dataset = FraudDataset(
        data_path='data/val.json',
        seq_length=config['seq_length']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = create_fraud_detector(
        input_dim=config['input_dim'],
        model_type='standard',
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers']
    )
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Calculate class weights for imbalanced data
    # This should be computed from your actual dataset
    class_weights = torch.tensor([1.0, 10.0])  # Adjust based on class distribution
    
    # Create trainer
    trainer = FraudDetectorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        class_weights=class_weights
    )
    
    # Train model
    print("\nStarting training...")
    trainer.train(
        num_epochs=config['num_epochs'],
        early_stopping_patience=15
    )
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
