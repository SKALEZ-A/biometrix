import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, DataLoader
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphAttentionNetwork(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_heads=4):
        super(GraphAttentionNetwork, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=0.6)
        self.conv3 = GATConv(hidden_channels * num_heads, num_classes, heads=1, concat=False, dropout=0.6)
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGENetwork(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GraphSAGENetwork, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, num_classes)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

class GNNFraudDetector:
    def __init__(self, model_type='GAT', num_features=64, hidden_channels=128, num_classes=2):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'GAT':
            self.model = GraphAttentionNetwork(num_features, hidden_channels, num_classes).to(self.device)
        elif model_type == 'GraphSAGE':
            self.model = GraphSAGENetwork(num_features, hidden_channels, num_classes).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)
        logger.info(f"Initialized {model_type} model on {self.device}")
