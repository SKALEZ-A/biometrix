import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from gnn_fraud_detector import GNNFraudDetector
from graph_builder import GraphBuilder
import json

class GNNTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GNNFraudDetector(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(out, batch.y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            
        return total_loss / len(train_loader), correct / total
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(out, batch.y)
                
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
                
        return total_loss / len(val_loader), correct / total
    
    def train(self, train_loader, val_loader, epochs):
        best_val_acc = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f'best_gnn_model.pth')
                
        return history
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

if __name__ == '__main__':
    config = {
        'input_dim': 64,
        'hidden_dim': 128,
        'output_dim': 2,
        'num_layers': 3,
        'dropout': 0.5,
        'learning_rate': 0.001,
        'weight_decay': 5e-4,
        'batch_size': 32,
        'epochs': 200
    }
    
    graph_builder = GraphBuilder()
    train_graphs = graph_builder.load_graphs('data/train_graphs.json')
    val_graphs = graph_builder.load_graphs('data/val_graphs.json')
    
    train_loader = DataLoader(train_graphs, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=config['batch_size'])
    
    trainer = GNNTrainer(config)
    history = trainer.train(train_loader, val_loader, config['epochs'])
