"""
GAN-based Synthetic Fraud Data Generator
Generates realistic synthetic fraud transactions for training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Generator(nn.Module):
    """
    Generator network for creating synthetic fraud transactions
    """
    
    def __init__(self, latent_dim: int = 100, output_dim: int = 50, hidden_dims: List[int] = [256, 512, 256]):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build generator network
        layers = []
        input_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Tanh())  # Normalize output to [-1, 1]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic transaction from noise vector
        
        Args:
            z: Noise vector of shape (batch_size, latent_dim)
            
        Returns:
            Generated transaction features of shape (batch_size, output_dim)
        """
        return self.model(z)
    
    def generate_samples(self, num_samples: int, device: str = 'cuda') -> torch.Tensor:
        """Generate multiple synthetic samples"""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.forward(z)
        return samples


class Discriminator(nn.Module):
    """
    Discriminator network for distinguishing real from fake transactions
    """
    
    def __init__(self, input_dim: int = 50, hidden_dims: List[int] = [256, 512, 256]):
        super(Discriminator, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify transaction as real or fake
        
        Args:
            x: Transaction features of shape (batch_size, input_dim)
            
        Returns:
            Probability of being real, shape (batch_size, 1)
        """
        return self.model(x)


class ConditionalGenerator(nn.Module):
    """
    Conditional GAN generator that can generate specific fraud types
    """
    
    def __init__(self, latent_dim: int = 100, num_classes: int = 10, 
                 output_dim: int = 50, hidden_dims: List[int] = [256, 512, 256]):
        super(ConditionalGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_dim = output_dim
        
        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        # Generator network
        layers = []
        input_dim = latent_dim * 2  # Concatenate noise and label embedding
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic transaction conditioned on fraud type
        
        Args:
            z: Noise vector of shape (batch_size, latent_dim)
            labels: Class labels of shape (batch_size,)
            
        Returns:
            Generated transaction features
        """
        # Embed labels
        label_embed = self.label_embedding(labels)
        
        # Concatenate noise and label embedding
        gen_input = torch.cat([z, label_embed], dim=1)
        
        return self.model(gen_input)
    
    def generate_fraud_type(self, fraud_type: int, num_samples: int, device: str = 'cuda') -> torch.Tensor:
        """Generate samples of a specific fraud type"""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            labels = torch.full((num_samples,), fraud_type, dtype=torch.long).to(device)
            samples = self.forward(z, labels)
        return samples


class ConditionalDiscriminator(nn.Module):
    """
    Conditional discriminator that considers fraud type
    """
    
    def __init__(self, input_dim: int = 50, num_classes: int = 10, 
                 hidden_dims: List[int] = [256, 512, 256]):
        super(ConditionalDiscriminator, self).__init__()
        
        self.num_classes = num_classes
        
        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, input_dim)
        
        # Discriminator network
        layers = []
        current_dim = input_dim * 2  # Concatenate features and label embedding
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Classify transaction as real or fake given fraud type
        
        Args:
            x: Transaction features
            labels: Class labels
            
        Returns:
            Probability of being real
        """
        label_embed = self.label_embedding(labels)
        disc_input = torch.cat([x, label_embed], dim=1)
        return self.model(disc_input)


class WassersteinGenerator(nn.Module):
    """
    Generator for Wasserstein GAN (WGAN) with improved training stability
    """
    
    def __init__(self, latent_dim: int = 100, output_dim: int = 50):
        super(WassersteinGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class WassersteinCritic(nn.Module):
    """
    Critic network for WGAN (replaces discriminator)
    """
    
    def __init__(self, input_dim: int = 50):
        super(WassersteinCritic, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FraudDataset(Dataset):
    """Dataset for real fraud transactions"""
    
    def __init__(self, data: np.ndarray, labels: Optional[np.ndarray] = None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels) if labels is not None else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]


class GANTrainer:
    """
    Trainer for GAN models
    """
    
    def __init__(self, generator: nn.Module, discriminator: nn.Module, 
                 device: str = 'cuda', gan_type: str = 'vanilla'):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.gan_type = gan_type
        
        # Optimizers
        self.g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        if gan_type == 'vanilla':
            self.criterion = nn.BCELoss()
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'd_real_acc': [],
            'd_fake_acc': []
        }
    
    def train_step(self, real_data: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Single training step"""
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)
        
        # Labels for real and fake data
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # ==================== Train Discriminator ====================
        self.d_optimizer.zero_grad()
        
        # Real data
        if labels is not None:
            labels = labels.to(self.device)
            d_real = self.discriminator(real_data, labels)
        else:
            d_real = self.discriminator(real_data)
        
        d_real_loss = self.criterion(d_real, real_labels)
        
        # Fake data
        z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
        if labels is not None:
            fake_data = self.generator(z, labels)
            d_fake = self.discriminator(fake_data.detach(), labels)
        else:
            fake_data = self.generator(z)
            d_fake = self.discriminator(fake_data.detach())
        
        d_fake_loss = self.criterion(d_fake, fake_labels)
        
        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # ==================== Train Generator ====================
        self.g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
        if labels is not None:
            fake_data = self.generator(z, labels)
            d_fake = self.discriminator(fake_data, labels)
        else:
            fake_data = self.generator(z)
            d_fake = self.discriminator(fake_data)
        
        g_loss = self.criterion(d_fake, real_labels)  # Generator wants discriminator to think fake is real
        g_loss.backward()
        self.g_optimizer.step()
        
        # Calculate accuracies
        d_real_acc = (d_real > 0.5).float().mean().item()
        d_fake_acc = (d_fake < 0.5).float().mean().item()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'd_real_acc': d_real_acc,
            'd_fake_acc': d_fake_acc
        }
    
    def train(self, dataloader: DataLoader, num_epochs: int = 100, 
              save_dir: str = './checkpoints'):
        """Full training loop"""
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting GAN training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_metrics = {
                'g_loss': 0.0,
                'd_loss': 0.0,
                'd_real_acc': 0.0,
                'd_fake_acc': 0.0
            }
            
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, tuple):
                    real_data, labels = batch
                else:
                    real_data = batch
                    labels = None
                
                metrics = self.train_step(real_data, labels)
                
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
            
            # Average metrics
            num_batches = len(dataloader)
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
                self.history[key].append(epoch_metrics[key])
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"G Loss: {epoch_metrics['g_loss']:.4f}, "
                f"D Loss: {epoch_metrics['d_loss']:.4f}, "
                f"D Real Acc: {epoch_metrics['d_real_acc']:.4f}, "
                f"D Fake Acc: {epoch_metrics['d_fake_acc']:.4f}"
            )
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"{save_dir}/checkpoint_epoch_{epoch+1}.pth", epoch)
        
        # Save final model
        self.save_checkpoint(f"{save_dir}/final_model.pth", num_epochs - 1)
        
        # Save training history
        with open(f"{save_dir}/training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info("Training completed")
    
    def save_checkpoint(self, filepath: str, epoch: int):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'history': self.history
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


class SyntheticDataGenerator:
    """
    High-level interface for generating synthetic fraud data
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.generator = self._load_generator(model_path)
        self.generator.eval()
    
    def _load_generator(self, model_path: str) -> nn.Module:
        """Load trained generator"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine generator type from checkpoint
        if 'num_classes' in checkpoint:
            generator = ConditionalGenerator()
        else:
            generator = Generator()
        
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.to(self.device)
        
        return generator
    
    def generate(self, num_samples: int, fraud_type: Optional[int] = None) -> np.ndarray:
        """
        Generate synthetic fraud transactions
        
        Args:
            num_samples: Number of samples to generate
            fraud_type: Optional fraud type for conditional generation
            
        Returns:
            Numpy array of synthetic transactions
        """
        self.generator.eval()
        
        with torch.no_grad():
            if fraud_type is not None and isinstance(self.generator, ConditionalGenerator):
                samples = self.generator.generate_fraud_type(fraud_type, num_samples, self.device)
            else:
                samples = self.generator.generate_samples(num_samples, self.device)
        
        return samples.cpu().numpy()
    
    def generate_balanced_dataset(self, samples_per_class: int, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate balanced dataset with equal samples per fraud type
        
        Returns:
            Tuple of (features, labels)
        """
        all_samples = []
        all_labels = []
        
        for fraud_type in range(num_classes):
            samples = self.generate(samples_per_class, fraud_type)
            labels = np.full(samples_per_class, fraud_type)
            
            all_samples.append(samples)
            all_labels.append(labels)
        
        features = np.vstack(all_samples)
        labels = np.concatenate(all_labels)
        
        # Shuffle
        indices = np.random.permutation(len(features))
        features = features[indices]
        labels = labels[indices]
        
        return features, labels
    
    def augment_dataset(self, real_data: np.ndarray, augmentation_ratio: float = 0.5) -> np.ndarray:
        """
        Augment real dataset with synthetic samples
        
        Args:
            real_data: Real fraud transactions
            augmentation_ratio: Ratio of synthetic to real samples
            
        Returns:
            Augmented dataset
        """
        num_synthetic = int(len(real_data) * augmentation_ratio)
        synthetic_data = self.generate(num_synthetic)
        
        augmented = np.vstack([real_data, synthetic_data])
        
        # Shuffle
        indices = np.random.permutation(len(augmented))
        augmented = augmented[indices]
        
        return augmented


def train_vanilla_gan(real_data: np.ndarray, latent_dim: int = 100, 
                     num_epochs: int = 100, batch_size: int = 64):
    """Train vanilla GAN"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create models
    generator = Generator(latent_dim=latent_dim, output_dim=real_data.shape[1])
    discriminator = Discriminator(input_dim=real_data.shape[1])
    
    # Create dataset and dataloader
    dataset = FraudDataset(real_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train
    trainer = GANTrainer(generator, discriminator, device=device)
    trainer.train(dataloader, num_epochs=num_epochs)
    
    return generator, discriminator


def train_conditional_gan(real_data: np.ndarray, labels: np.ndarray, 
                         num_classes: int, latent_dim: int = 100,
                         num_epochs: int = 100, batch_size: int = 64):
    """Train conditional GAN"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create models
    generator = ConditionalGenerator(latent_dim=latent_dim, num_classes=num_classes,
                                    output_dim=real_data.shape[1])
    discriminator = ConditionalDiscriminator(input_dim=real_data.shape[1], 
                                            num_classes=num_classes)
    
    # Create dataset and dataloader
    dataset = FraudDataset(real_data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train
    trainer = GANTrainer(generator, discriminator, device=device)
    trainer.train(dataloader, num_epochs=num_epochs)
    
    return generator, discriminator


if __name__ == "__main__":
    # Example usage
    logger.info("GAN Synthetic Fraud Generator initialized")
    
    # Generate sample data
    # real_data = np.random.randn(1000, 50)
    # labels = np.random.randint(0, 10, 1000)
    
    # Train conditional GAN
    # generator, discriminator = train_conditional_gan(real_data, labels, num_classes=10)
    
    # Generate synthetic data
    # synthetic_gen = SyntheticDataGenerator('checkpoints/final_model.pth')
    # synthetic_data = synthetic_gen.generate(100, fraud_type=5)
    
    logger.info("Ready to generate synthetic fraud data")
