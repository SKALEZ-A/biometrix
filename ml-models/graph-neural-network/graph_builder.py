import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class FraudGraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()
        
    def build_transaction_graph(self, transactions: pd.DataFrame) -> nx.Graph:
        """Build graph from transaction data"""
        self.graph.clear()
        
        # Add nodes
        self._add_user_nodes(transactions)
        self._add_merchant_nodes(transactions)
        self._add_device_nodes(transactions)
        
        # Add edges
        self._add_transaction_edges(transactions)
        self._add_device_edges(transactions)
        self._add_location_edges(transactions)
        
        return self.graph
    
    def _add_user_nodes(self, transactions: pd.DataFrame):
        """Add user nodes to graph"""
        users = transactions['user_id'].unique()
        for user in users:
            user_txns = transactions[transactions['user_id'] == user]
            self.graph.add_node(
                f'user_{user}',
                node_type='user',
                txn_count=len(user_txns),
                avg_amount=user_txns['amount'].mean(),
                fraud_rate=user_txns['is_fraud'].mean() if 'is_fraud' in user_txns else 0
            )
    
    def _add_merchant_nodes(self, transactions: pd.DataFrame):
        """Add merchant nodes to graph"""
        merchants = transactions['merchant_id'].unique()
        for merchant in merchants:
            merchant_txns = transactions[transactions['merchant_id'] == merchant]
            self.graph.add_node(
                f'merchant_{merchant}',
                node_type='merchant',
                txn_count=len(merchant_txns),
                avg_amount=merchant_txns['amount'].mean(),
                fraud_rate=merchant_txns['is_fraud'].mean() if 'is_fraud' in merchant_txns else 0
            )
    
    def _add_device_nodes(self, transactions: pd.DataFrame):
        """Add device nodes to graph"""
        devices = transactions['device_fingerprint'].unique()
        for device in devices:
            device_txns = transactions[transactions['device_fingerprint'] == device]
            self.graph.add_node(
                f'device_{device}',
                node_type='device',
                user_count=device_txns['user_id'].nunique(),
                txn_count=len(device_txns)
            )
    
    def _add_transaction_edges(self, transactions: pd.DataFrame):
        """Add transaction edges between users and merchants"""
        for _, txn in transactions.iterrows():
            self.graph.add_edge(
                f'user_{txn["user_id"]}',
                f'merchant_{txn["merchant_id"]}',
                weight=txn['amount'],
                timestamp=txn['timestamp']
            )
    
    def _add_device_edges(self, transactions: pd.DataFrame):
        """Add edges between users and devices"""
        for _, txn in transactions.iterrows():
            self.graph.add_edge(
                f'user_{txn["user_id"]}',
                f'device_{txn["device_fingerprint"]}',
                edge_type='uses_device'
            )
    
    def _add_location_edges(self, transactions: pd.DataFrame):
        """Add edges based on location proximity"""
        # Group by location and find users in same location
        location_groups = transactions.groupby('location')['user_id'].apply(list)
        
        for location, users in location_groups.items():
            if len(users) > 1:
                for i in range(len(users)):
                    for j in range(i + 1, len(users)):
                        if not self.graph.has_edge(f'user_{users[i]}', f'user_{users[j]}'):
                            self.graph.add_edge(
                                f'user_{users[i]}',
                                f'user_{users[j]}',
                                edge_type='same_location',
                                location=location
                            )
    
    def extract_subgraph(self, center_node: str, radius: int = 2) -> nx.Graph:
        """Extract subgraph around a node"""
        nodes = nx.single_source_shortest_path_length(self.graph, center_node, cutoff=radius)
        return self.graph.subgraph(nodes.keys())
    
    def calculate_node_features(self) -> Dict[str, np.ndarray]:
        """Calculate features for each node"""
        features = {}
        
        for node in self.graph.nodes():
            node_features = [
                self.graph.degree(node),
                nx.clustering(self.graph, node),
                nx.pagerank(self.graph)[node]
            ]
            features[node] = np.array(node_features)
        
        return features
