import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model_architecture import AttentionFraudDetector
import json
from datetime import datetime

class FraudDataset(Dataset):
    def __init__(self, data_path, sequence_length=50):
        self.data = self.load_data(data_path)
        self.sequence_length = sequence_length
        
    def load_data(self, path):
        with open(path, 'r') as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.tensor(item['features'], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.long)
        return features, label

class AttentionFraudTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AttentionFraudDetector(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers']
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5
        )
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (features, labels) in enumerate(dataloader):
            features, labels = features.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return total_loss / len(dataloader), 100. * correct / total
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in dataloader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return total_loss / len(dataloader), 100. * correct / total
    
    def train(self, train_loader, val_loader, epochs):
        best_val_acc = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f'best_model_epoch_{epoch+1}.pth')
                
        return history
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

if __name__ == '__main__':
    config = {
        'input_dim': 128,
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 4,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 64,
        'epochs': 100
    }
    
    train_dataset = FraudDataset('data/train.json')
    val_dataset = FraudDataset('data/val.json')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    trainer = AttentionFraudTrainer(config)
    history = trainer.train(train_loader, val_loader, config['epochs'])
