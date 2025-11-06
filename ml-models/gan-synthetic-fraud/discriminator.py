"""
Advanced Discriminator Architectures for GAN Training
Multiple discriminator variants for improved fraud detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np

class SpectralNorm(nn.Module):
    """
    Spectral Normalization for stable GAN training
    """
    
    def __init__(self, module: nn.Module, name: str = 'weight', power_iterations: int = 1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        
        if not self._made_params():
            self._make_params()
    
    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False
    
    def _make_params(self):
        w = getattr(self.module, self.name)
        
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)
        w_bar = nn.Parameter(w.data)
        
        del self.module._parameters[self.name]
        
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)
    
    def _l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)
    
    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
    
    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self._l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = self._l2normalize(torch.mv(w.view(height, -1).data, v.data))
        
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for discriminator
    """
    
    def __init__(self, in_channels: int):
        super(SelfAttention, self).__init__()
        
        self.in_channels = in_channels
        
        self.query = nn.Linear(in_channels, in_channels // 8)
        self.key = nn.Linear(in_channels, in_channels // 8)
        self.value = nn.Linear(in_channels, in_channels)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels)
        """
        batch_size = x.size(0)
        
        # Compute query, key, value
        query = self.query(x).view(batch_size, -1, 1)
        key = self.key(x).view(batch_size, -1, 1)
        value = self.value(x).view(batch_size, -1, 1)
        
        # Attention scores
        attention = torch.bmm(query.transpose(1, 2), key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.transpose(1, 2))
        out = out.view(batch_size, -1)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        return out


class MinibatchDiscrimination(nn.Module):
    """
    Minibatch discrimination to prevent mode collapse
    """
    
    def __init__(self, in_features: int, out_features: int, kernel_dims: int = 5):
        super(MinibatchDiscrimination, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)
        """
        batch_size = x.size(0)
        
        # Compute activations
        M = x.mm(self.T.view(self.in_features, -1))
        M = M.view(batch_size, self.out_features, self.kernel_dims)
        
        # Compute L1 distances between samples
        M_expanded = M.unsqueeze(0)
        M_tiled = M.unsqueeze(1)
        
        diffs = torch.abs(M_expanded - M_tiled).sum(3)
        c = torch.exp(-diffs).sum(1) - 1  # Exclude self
        
        return torch.cat([x, c], dim=1)


class AdvancedDiscriminator(nn.Module):
    """
    Advanced discriminator with spectral normalization and self-attention
    """
    
    def __init__(self, input_dim: int = 50, hidden_dims: List[int] = [256, 512, 256],
                 use_spectral_norm: bool = True, use_attention: bool = True):
        super(AdvancedDiscriminator, self).__init__()
        
        self.use_attention = use_attention
        
        layers = []
        current_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(current_dim, hidden_dim)
            
            if use_spectral_norm:
                linear = SpectralNorm(linear)
            
            layers.append(linear)
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
            
            # Add self-attention after first hidden layer
            if use_attention and i == 0:
                layers.append(SelfAttention(hidden_dim))
            
            current_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        
        # Output layer
        self.output = nn.Linear(current_dim, 1)
        if use_spectral_norm:
            self.output = SpectralNorm(self.output)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        output = self.output(features)
        return torch.sigmoid(output)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for better feature extraction
    """
    
    def __init__(self, input_dim: int = 50, num_scales: int = 3):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()
        
        for i in range(num_scales):
            scale_dim = input_dim // (2 ** i)
            disc = nn.Sequential(
                nn.Linear(scale_dim, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            self.discriminators.append(disc)
        
        self.downsample = nn.AvgPool1d(2, stride=2)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns predictions at multiple scales
        """
        outputs = []
        current_x = x
        
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                # Downsample for next scale
                current_x = current_x.unsqueeze(1)
                current_x = self.downsample(current_x)
                current_x = current_x.squeeze(1)
            
            output = disc(current_x)
            outputs.append(output)
        
        return outputs


class ProjectionDiscriminator(nn.Module):
    """
    Projection discriminator for conditional GAN
    """
    
    def __init__(self, input_dim: int = 50, num_classes: int = 10, 
                 hidden_dims: List[int] = [256, 512, 256]):
        super(ProjectionDiscriminator, self).__init__()
        
        # Feature extraction
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        
        # Projection layers
        self.projection = nn.Linear(current_dim, 1)
        self.embedding = nn.Embedding(num_classes, current_dim)
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Projection-based conditional discrimination
        """
        features = self.features(x)
        
        # Standard discrimination
        output = self.projection(features)
        
        # Add projection of label embedding
        label_embed = self.embedding(labels)
        projection = (features * label_embed).sum(dim=1, keepdim=True)
        
        output = output + projection
        
        return torch.sigmoid(output)


class AuxiliaryClassifierDiscriminator(nn.Module):
    """
    AC-GAN discriminator with auxiliary classifier
    """
    
    def __init__(self, input_dim: int = 50, num_classes: int = 10,
                 hidden_dims: List[int] = [256, 512, 256]):
        super(AuxiliaryClassifierDiscriminator, self).__init__()
        
        # Shared feature extraction
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        
        # Real/fake discrimination head
        self.discriminator = nn.Sequential(
            nn.Linear(current_dim, 1),
            nn.Sigmoid()
        )
        
        # Class classification head
        self.classifier = nn.Sequential(
            nn.Linear(current_dim, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns both real/fake prediction and class prediction
        """
        features = self.features(x)
        
        real_fake = self.discriminator(features)
        class_pred = self.classifier(features)
        
        return real_fake, class_pred


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for local discrimination
    """
    
    def __init__(self, input_dim: int = 50, patch_size: int = 10):
        super(PatchDiscriminator, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = input_dim // patch_size
        
        # Discriminator for each patch
        self.patch_disc = nn.Sequential(
            nn.Linear(patch_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Global discriminator
        self.global_disc = nn.Sequential(
            nn.Linear(self.num_patches, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Discriminate on patches and aggregate
        """
        batch_size = x.size(0)
        
        # Split into patches
        patches = x.view(batch_size, self.num_patches, self.patch_size)
        
        # Discriminate each patch
        patch_outputs = []
        for i in range(self.num_patches):
            patch = patches[:, i, :]
            output = self.patch_disc(patch)
            patch_outputs.append(output)
        
        # Stack patch outputs
        patch_outputs = torch.cat(patch_outputs, dim=1)
        
        # Global discrimination
        global_output = self.global_disc(patch_outputs)
        
        return global_output


class EnsembleDiscriminator(nn.Module):
    """
    Ensemble of multiple discriminators for robust detection
    """
    
    def __init__(self, input_dim: int = 50, num_discriminators: int = 3):
        super(EnsembleDiscriminator, self).__init__()
        
        self.discriminators = nn.ModuleList()
        
        for i in range(num_discriminators):
            # Create discriminator with different architecture
            hidden_dims = [256 + i * 64, 512 + i * 64, 256 + i * 64]
            disc = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(hidden_dims[1], hidden_dims[2]),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(hidden_dims[2], 1),
                nn.Sigmoid()
            )
            self.discriminators.append(disc)
        
        # Learnable weights for ensemble
        self.weights = nn.Parameter(torch.ones(num_discriminators) / num_discriminators)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Weighted ensemble prediction
        """
        outputs = []
        
        for disc in self.discriminators:
            output = disc(x)
            outputs.append(output)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=2)
        
        # Weighted average
        weights = F.softmax(self.weights, dim=0)
        ensemble_output = (outputs * weights.view(1, 1, -1)).sum(dim=2)
        
        return ensemble_output


class ResidualDiscriminator(nn.Module):
    """
    Discriminator with residual connections
    """
    
    def __init__(self, input_dim: int = 50, hidden_dim: int = 256, num_blocks: int = 3):
        super(ResidualDiscriminator, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = ResidualBlock(hidden_dim)
            self.blocks.append(block)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = F.leaky_relu(x, 0.2)
        
        for block in self.blocks:
            x = block(x)
        
        output = self.output(x)
        return output


class ResidualBlock(nn.Module):
    """Residual block for discriminator"""
    
    def __init__(self, dim: int):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x + self.block(x), 0.2)


class GradientPenaltyDiscriminator(nn.Module):
    """
    Discriminator with gradient penalty for WGAN-GP
    """
    
    def __init__(self, input_dim: int = 50, hidden_dims: List[int] = [256, 512, 256]):
        super(GradientPenaltyDiscriminator, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def compute_gradient_penalty(self, real_data: torch.Tensor, 
                                fake_data: torch.Tensor, lambda_gp: float = 10.0) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP
        """
        batch_size = real_data.size(0)
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1).to(real_data.device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        
        # Compute discriminator output
        disc_interpolates = self.forward(interpolates)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
        
        return gradient_penalty


def create_discriminator(disc_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create discriminators
    """
    
    discriminators = {
        'basic': AdvancedDiscriminator,
        'multiscale': MultiScaleDiscriminator,
        'projection': ProjectionDiscriminator,
        'ac_gan': AuxiliaryClassifierDiscriminator,
        'patch': PatchDiscriminator,
        'ensemble': EnsembleDiscriminator,
        'residual': ResidualDiscriminator,
        'wgan_gp': GradientPenaltyDiscriminator
    }
    
    if disc_type not in discriminators:
        raise ValueError(f"Unknown discriminator type: {disc_type}")
    
    return discriminators[disc_type](**kwargs)


if __name__ == "__main__":
    # Test discriminators
    batch_size = 32
    input_dim = 50
    
    x = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, 10, (batch_size,))
    
    # Test basic discriminator
    disc = AdvancedDiscriminator(input_dim=input_dim)
    output = disc(x)
    print(f"Basic discriminator output shape: {output.shape}")
    
    # Test projection discriminator
    proj_disc = ProjectionDiscriminator(input_dim=input_dim, num_classes=10)
    output = proj_disc(x, labels)
    print(f"Projection discriminator output shape: {output.shape}")
    
    # Test AC-GAN discriminator
    ac_disc = AuxiliaryClassifierDiscriminator(input_dim=input_dim, num_classes=10)
    real_fake, class_pred = ac_disc(x)
    print(f"AC-GAN outputs - Real/Fake: {real_fake.shape}, Class: {class_pred.shape}")
    
    print("All discriminators tested successfully")
