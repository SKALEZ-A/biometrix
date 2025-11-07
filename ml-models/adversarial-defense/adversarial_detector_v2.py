import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import logging
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
from datetime import datetime

class AdversarialBiometricDetectorV2(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_classes: int = 2):
        super(AdversarialBiometricDetectorV2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Encoder for feature extraction
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head with adversarial robustness
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
        
        # Adversarial discriminator (for GAN training)
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger('AdversarialDetectorV2')
    
    def forward(self, x: torch.Tensor, adversarial_mode: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional adversarial training."""
        encoded = self.encoder(x)
        class_output = self.classifier(encoded)
        
        if adversarial_mode:
            disc_output = self.discriminator(encoded)
            return class_output, disc_output
        return class_output, None
    
    def adversarial_loss(self, real_features: torch.Tensor, fake_features: torch.Tensor, 
                        class_labels: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        """Compute combined classification and adversarial loss."""
        # Classification loss
        class_criterion = nn.CrossEntropyLoss()
        class_loss = class_criterion(class_labels, self.discriminator(real_features))
        
        # GAN loss for robustness
        real_labels = torch.ones(real_features.size(0), 1).to(real_features.device)
        fake_labels = torch.zeros(fake_features.size(0), 1).to(fake_features.device)
        
        d_real_loss = nn.BCELoss()(self.discriminator(real_features), real_labels)
        d_fake_loss = nn.BCELoss()(self.discriminator(fake_features.detach()), fake_labels)
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        g_loss = nn.BCELoss()(self.discriminator(fake_features), real_labels)
        
        # Combined robust loss
        total_loss = class_loss + alpha * (d_loss + g_loss)
        return total_loss

class AdversarialTrainer:
    def __init__(self, model: AdversarialBiometricDetectorV2, lr: float = 0.001, epochs: int = 100):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.logger = logging.getLogger('AdversarialTrainer')
    
    def generate_adversarial_samples(self, X_real: np.ndarray, y_real: np.ndarray, 
                                   epsilon: float = 0.1, n_attacks: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate adversarial examples using FGSM-like perturbation."""
        self.logger.info("Generating adversarial samples")
        X_adv = X_real.copy().astype(np.float32)
        y_adv = y_real.copy()
        
        for i in range(n_attacks):
            # Simple perturbation (in production, use more sophisticated attacks)
            perturbation = np.random.normal(0, epsilon, X_real.shape[1])
            X_adv[i] = np.clip(X_real[i] + perturbation, 0, 1)
            if np.random.random() < 0.3:  # Flip some labels for evasion
                y_adv[i] = 1 - y_adv[i]
        
        return X_adv, y_adv
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train the model with adversarial robustness."""
        # Prepare data
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        best_val_acc = 0.0
        history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Generate batch adversarial samples
                batch_x_adv, batch_y_adv = self.generate_adversarial_samples(
                    batch_x.cpu().numpy(), batch_y.cpu().numpy(), epsilon=0.05, n_attacks=len(batch_x)
                )
                batch_x_adv = torch.FloatTensor(batch_x_adv).to(self.device)
                batch_y_adv = torch.LongTensor(batch_y_adv).to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                class_out_real, _ = self.model(batch_x, adversarial_mode=True)
                class_out_adv, disc_real = self.model(batch_x_adv, adversarial_mode=True)
                
                # Losses
                class_loss_real = nn.CrossEntropyLoss()(class_out_real, batch_y)
                class_loss_adv = nn.CrossEntropyLoss()(class_out_adv, batch_y_adv)
                
                # Adversarial component
                fake_features = self.model.encoder(batch_x_adv).detach()
                adv_loss = self.model.adversarial_loss(
                    self.model.encoder(batch_x).detach(), fake_features, batch_y, alpha=0.1
                )
                
                loss = (class_loss_real + class_loss_adv) / 2 + adv_loss
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            self.scheduler.step()
            
            # Validation
            val_acc, val_f1 = self.evaluate(val_loader)
            history['train_loss'].append(total_loss / len(train_loader))
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_adversarial_model.pth')
            
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}: Loss={total_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
        
        return history
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs, _ = self.model(batch_x)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return acc, f1
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data."""
        self.model.eval()
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for batch_x, in loader:
                batch_x = batch_x.to(self.device)
                outputs, _ = self.model(batch_x)
                probs = nn.Softmax(dim=1)(outputs)[:, 1].cpu().numpy()  # Fraud probability
                all_probs.extend(probs)
        
        return np.array(all_probs)

# Example usage
if __name__ == "__main__":
    # Synthetic biometric data (e.g., facial embeddings)
    np.random.seed(42)
    n_samples = 5000
    input_dim = 512
    X = np.random.rand(n_samples, input_dim)
    y = np.random.randint(0, 2, n_samples)  # 0: genuine, 1: fraud/spoof
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train
    model = AdversarialBiometricDetectorV2(input_dim=input_dim)
    trainer = AdversarialTrainer(model, epochs=50)
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    val_acc, val_f1 = trainer.evaluate(DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=64))
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    print(f"Final Validation F1: {val_f1:.4f}")
    
    # Test prediction
    test_probs = trainer.predict(X_val[:10])
    print("Sample Fraud Probabilities:", test_probs)
