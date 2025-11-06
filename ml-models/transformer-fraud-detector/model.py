import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from typing import Dict, List, Tuple, Optional

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadAttentionFraudDetector(nn.Module):
    """
    Transformer-based fraud detection model with multi-head attention
    for analyzing transaction sequences and patterns
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 100,
        num_classes: int = 2
    ):
        super(MultiHeadAttentionFraudDetector, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes)
        )
        
        # Attention weights for interpretability
        self.attention_weights = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            src_mask: Attention mask
            src_key_padding_mask: Padding mask
            
        Returns:
            Tuple of (logits, attention_weights)
        """
        # Embed input
        x = self.input_embedding(x) * np.sqrt(self.d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_length, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_length, d_model)
        
        # Apply transformer encoder
        encoded = self.transformer_encoder(
            x,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Global average pooling over sequence dimension
        pooled = torch.mean(encoded, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits, encoded
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights for interpretability"""
        return self.attention_weights


class TransactionSequenceEncoder(nn.Module):
    """
    Encodes transaction sequences with temporal and categorical features
    """
    
    def __init__(
        self,
        numerical_features: int,
        categorical_features: Dict[str, int],
        embedding_dim: int = 64,
        hidden_dim: int = 256
    ):
        super(TransactionSequenceEncoder, self).__init__()
        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(num_categories, embedding_dim)
            for name, num_categories in categorical_features.items()
        })
        
        # Calculate total input dimension
        total_categorical_dim = len(categorical_features) * embedding_dim
        self.input_dim = numerical_features + total_categorical_dim
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    
    def forward(
        self,
        numerical_features: torch.Tensor,
        categorical_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Encode transaction features
        
        Args:
            numerical_features: Tensor of shape (batch_size, seq_length, num_features)
            categorical_features: Dict of tensors with shape (batch_size, seq_length)
            
        Returns:
            Encoded features of shape (batch_size, seq_length, hidden_dim)
        """
        # Embed categorical features
        embedded_cats = []
        for name, embedding_layer in self.embeddings.items():
            if name in categorical_features:
                embedded = embedding_layer(categorical_features[name])
                embedded_cats.append(embedded)
        
        # Concatenate all features
        if embedded_cats:
            categorical_tensor = torch.cat(embedded_cats, dim=-1)
            combined = torch.cat([numerical_features, categorical_tensor], dim=-1)
        else:
            combined = numerical_features
        
        # Transform features
        encoded = self.feature_transform(combined)
        
        return encoded


class HierarchicalFraudDetector(nn.Module):
    """
    Hierarchical transformer model for fraud detection
    Processes transactions at multiple levels: individual, session, and user
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super(HierarchicalFraudDetector, self).__init__()
        
        # Transaction-level encoder
        self.transaction_encoder = MultiHeadAttentionFraudDetector(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dropout=dropout
        )
        
        # Session-level encoder
        self.session_encoder = MultiHeadAttentionFraudDetector(
            input_dim=d_model,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers // 2,
            dropout=dropout
        )
        
        # User-level aggregation
        self.user_aggregator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU()
        )
        
        # Final classifier
        self.classifier = nn.Linear(d_model // 2, 2)
    
    def forward(
        self,
        transactions: torch.Tensor,
        session_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through hierarchical model
        
        Args:
            transactions: Transaction features (batch, seq_len, features)
            session_mask: Mask indicating session boundaries
            
        Returns:
            Fraud probability logits
        """
        # Encode transactions
        trans_logits, trans_encoded = self.transaction_encoder(transactions)
        
        # Encode sessions
        session_logits, session_encoded = self.session_encoder(trans_encoded)
        
        # Aggregate at user level
        trans_pooled = torch.mean(trans_encoded, dim=1)
        session_pooled = torch.mean(session_encoded, dim=1)
        
        user_features = torch.cat([trans_pooled, session_pooled], dim=-1)
        user_encoded = self.user_aggregator(user_features)
        
        # Final classification
        logits = self.classifier(user_encoded)
        
        return logits


class FraudDetectionLoss(nn.Module):
    """
    Custom loss function for fraud detection with class imbalance handling
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super(FraudDetectionLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for fraud detection
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


def create_fraud_detector(
    input_dim: int,
    model_type: str = 'standard',
    **kwargs
) -> nn.Module:
    """
    Factory function to create fraud detection models
    
    Args:
        input_dim: Input feature dimension
        model_type: Type of model ('standard', 'hierarchical')
        **kwargs: Additional model parameters
        
    Returns:
        Fraud detection model
    """
    if model_type == 'standard':
        return MultiHeadAttentionFraudDetector(input_dim=input_dim, **kwargs)
    elif model_type == 'hierarchical':
        return HierarchicalFraudDetector(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    batch_size = 32
    seq_length = 50
    input_dim = 128
    
    # Create model
    model = create_fraud_detector(
        input_dim=input_dim,
        model_type='standard',
        d_model=256,
        nhead=8,
        num_encoder_layers=4
    )
    
    # Create sample input
    x = torch.randn(batch_size, seq_length, input_dim)
    
    # Forward pass
    logits, encoded = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Encoded features shape: {encoded.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
