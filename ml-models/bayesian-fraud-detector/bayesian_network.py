import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json

class BayesianNode:
    """Represents a node in the Bayesian Network"""
    
    def __init__(self, name: str, states: List[str], parents: List[str] = None):
        self.name = name
        self.states = states
        self.parents = parents or []
        self.cpt = {}  # Conditional Probability Table
        self.evidence = None
        
    def set_cpt(self, cpt: Dict):
        """Set the conditional probability table"""
        self.cpt = cpt
        
    def get_probability(self, state: str, parent_states: Dict = None) -> float:
        """Get probability of a state given parent states"""
        if not self.parents:
            return self.cpt.get(state, 0.0)
        
        parent_key = tuple(parent_states.get(p, None) for p in self.parents)
        return self.cpt.get(parent_key, {}).get(state, 0.0)
    
    def set_evidence(self, state: str):
        """Set evidence for this node"""
        self.evidence = state
        
    def clear_evidence(self):
        """Clear evidence from this node"""
        self.evidence = None


class BayesianFraudDetector:
    """Bayesian Network for Fraud Detection"""
    
    def __init__(self):
        self.nodes = {}
        self.topology = []
        self.initialize_network()
        
    def initialize_network(self):
        """Initialize the Bayesian Network structure for fraud detection"""
        
        # Define nodes
        self.add_node('transaction_amount', ['low', 'medium', 'high', 'very_high'])
        self.add_node('transaction_frequency', ['low', 'normal', 'high', 'very_high'])
        self.add_node('location_change', ['no', 'same_city', 'same_country', 'international'])
        self.add_node('device_change', ['no', 'yes'])
        self.add_node('time_of_day', ['night', 'morning', 'afternoon', 'evening'])
        self.add_node('merchant_category', ['low_risk', 'medium_risk', 'high_risk'])
        self.add_node('user_history', ['new', 'regular', 'trusted'])
        
        # Intermediate nodes
        self.add_node('behavioral_anomaly', ['no', 'minor', 'major'], 
                     ['transaction_frequency', 'location_change', 'device_change'])
        self.add_node('transaction_risk', ['low', 'medium', 'high'], 
                     ['transaction_amount', 'merchant_category', 'time_of_day'])
        self.add_node('user_trust_score', ['low', 'medium', 'high'], 
                     ['user_history', 'behavioral_anomaly'])
        
        # Final fraud node
        self.add_node('fraud', ['no', 'yes'], 
                     ['transaction_risk', 'user_trust_score', 'behavioral_anomaly'])
        
        # Set conditional probability tables
        self.set_prior_probabilities()
        self.set_conditional_probabilities()
        
    def add_node(self, name: str, states: List[str], parents: List[str] = None):
        """Add a node to the network"""
        node = BayesianNode(name, states, parents)
        self.nodes[name] = node
        self.topology.append(name)
        
    def set_prior_probabilities(self):
        """Set prior probabilities for root nodes"""
        
        # Transaction amount priors
        self.nodes['transaction_amount'].set_cpt({
            'low': 0.50,
            'medium': 0.30,
            'high': 0.15,
            'very_high': 0.05
        })
        
        # Transaction frequency priors
        self.nodes['transaction_frequency'].set_cpt({
            'low': 0.20,
            'normal': 0.60,
            'high': 0.15,
            'very_high': 0.05
        })
        
        # Location change priors
        self.nodes['location_change'].set_cpt({
            'no': 0.70,
            'same_city': 0.20,
            'same_country': 0.08,
            'international': 0.02
        })
        
        # Device change priors
        self.nodes['device_change'].set_cpt({
            'no': 0.85,
            'yes': 0.15
        })
        
        # Time of day priors
        self.nodes['time_of_day'].set_cpt({
            'night': 0.15,
            'morning': 0.25,
            'afternoon': 0.35,
            'evening': 0.25
        })
        
        # Merchant category priors
        self.nodes['merchant_category'].set_cpt({
            'low_risk': 0.70,
            'medium_risk': 0.25,
            'high_risk': 0.05
        })
        
        # User history priors
        self.nodes['user_history'].set_cpt({
            'new': 0.10,
            'regular': 0.70,
            'trusted': 0.20
        })
        
    def set_conditional_probabilities(self):
        """Set conditional probability tables for dependent nodes"""
        
        # Behavioral anomaly CPT
        behavioral_cpt = {}
        for freq in ['low', 'normal', 'high', 'very_high']:
            for loc in ['no', 'same_city', 'same_country', 'international']:
                for dev in ['no', 'yes']:
                    key = (freq, loc, dev)
                    
                    # Calculate anomaly probability based on factors
                    anomaly_score = 0
                    if freq in ['high', 'very_high']:
                        anomaly_score += 0.3
                    if loc in ['same_country', 'international']:
                        anomaly_score += 0.4
                    if dev == 'yes':
                        anomaly_score += 0.3
                    
                    if anomaly_score >= 0.7:
                        behavioral_cpt[key] = {'no': 0.1, 'minor': 0.3, 'major': 0.6}
                    elif anomaly_score >= 0.4:
                        behavioral_cpt[key] = {'no': 0.3, 'minor': 0.5, 'major': 0.2}
                    else:
                        behavioral_cpt[key] = {'no': 0.8, 'minor': 0.15, 'major': 0.05}
        
        self.nodes['behavioral_anomaly'].set_cpt(behavioral_cpt)
        
        # Transaction risk CPT
        transaction_risk_cpt = {}
        for amount in ['low', 'medium', 'high', 'very_high']:
            for merchant in ['low_risk', 'medium_risk', 'high_risk']:
                for time in ['night', 'morning', 'afternoon', 'evening']:
                    key = (amount, merchant, time)
                    
                    risk_score = 0
                    if amount in ['high', 'very_high']:
                        risk_score += 0.4
                    if merchant == 'high_risk':
                        risk_score += 0.4
                    elif merchant == 'medium_risk':
                        risk_score += 0.2
                    if time == 'night':
                        risk_score += 0.2
                    
                    if risk_score >= 0.7:
                        transaction_risk_cpt[key] = {'low': 0.1, 'medium': 0.3, 'high': 0.6}
                    elif risk_score >= 0.4:
                        transaction_risk_cpt[key] = {'low': 0.3, 'medium': 0.5, 'high': 0.2}
                    else:
                        transaction_risk_cpt[key] = {'low': 0.7, 'medium': 0.25, 'high': 0.05}
        
        self.nodes['transaction_risk'].set_cpt(transaction_risk_cpt)
        
        # User trust score CPT
        trust_cpt = {}
        for history in ['new', 'regular', 'trusted']:
            for anomaly in ['no', 'minor', 'major']:
                key = (history, anomaly)
                
                if history == 'trusted' and anomaly == 'no':
                    trust_cpt[key] = {'low': 0.05, 'medium': 0.15, 'high': 0.80}
                elif history == 'trusted' and anomaly == 'minor':
                    trust_cpt[key] = {'low': 0.10, 'medium': 0.40, 'high': 0.50}
                elif history == 'trusted' and anomaly == 'major':
                    trust_cpt[key] = {'low': 0.40, 'medium': 0.40, 'high': 0.20}
                elif history == 'regular' and anomaly == 'no':
                    trust_cpt[key] = {'low': 0.10, 'medium': 0.40, 'high': 0.50}
                elif history == 'regular' and anomaly == 'minor':
                    trust_cpt[key] = {'low': 0.30, 'medium': 0.50, 'high': 0.20}
                elif history == 'regular' and anomaly == 'major':
                    trust_cpt[key] = {'low': 0.70, 'medium': 0.25, 'high': 0.05}
                elif history == 'new' and anomaly == 'no':
                    trust_cpt[key] = {'low': 0.40, 'medium': 0.40, 'high': 0.20}
                elif history == 'new' and anomaly == 'minor':
                    trust_cpt[key] = {'low': 0.60, 'medium': 0.30, 'high': 0.10}
                else:  # new and major
                    trust_cpt[key] = {'low': 0.85, 'medium': 0.12, 'high': 0.03}
        
        self.nodes['user_trust_score'].set_cpt(trust_cpt)
        
        # Fraud CPT
        fraud_cpt = {}
        for trans_risk in ['low', 'medium', 'high']:
            for trust in ['low', 'medium', 'high']:
                for anomaly in ['no', 'minor', 'major']:
                    key = (trans_risk, trust, anomaly)
                    
                    fraud_score = 0
                    if trans_risk == 'high':
                        fraud_score += 0.4
                    elif trans_risk == 'medium':
                        fraud_score += 0.2
                    
                    if trust == 'low':
                        fraud_score += 0.4
                    elif trust == 'medium':
                        fraud_score += 0.2
                    
                    if anomaly == 'major':
                        fraud_score += 0.3
                    elif anomaly == 'minor':
                        fraud_score += 0.15
                    
                    fraud_prob = min(0.95, fraud_score)
                    fraud_cpt[key] = {'no': 1 - fraud_prob, 'yes': fraud_prob}
        
        self.nodes['fraud'].set_cpt(fraud_cpt)
        
    def predict(self, evidence: Dict[str, str]) -> Dict[str, float]:
        """
        Predict fraud probability given evidence
        
        Args:
            evidence: Dictionary of observed node states
            
        Returns:
            Dictionary with fraud probabilities
        """
        # Set evidence
        for node_name, state in evidence.items():
            if node_name in self.nodes:
                self.nodes[node_name].set_evidence(state)
        
        # Perform inference using variable elimination
        fraud_probs = self.variable_elimination('fraud')
        
        # Clear evidence
        for node in self.nodes.values():
            node.clear_evidence()
        
        return fraud_probs
    
    def variable_elimination(self, query_var: str) -> Dict[str, float]:
        """
        Perform variable elimination for inference
        
        Args:
            query_var: Variable to query
            
        Returns:
            Probability distribution over query variable
        """
        # Simplified variable elimination
        # In production, use proper belief propagation
        
        # For now, use forward sampling approximation
        num_samples = 10000
        samples = self.forward_sampling(num_samples)
        
        # Count occurrences
        counts = defaultdict(int)
        total = 0
        
        for sample in samples:
            if query_var in sample:
                counts[sample[query_var]] += 1
                total += 1
        
        # Normalize
        probs = {state: counts[state] / total if total > 0 else 0 
                for state in self.nodes[query_var].states}
        
        return probs
    
    def forward_sampling(self, num_samples: int) -> List[Dict[str, str]]:
        """Generate samples using forward sampling"""
        samples = []
        
        for _ in range(num_samples):
            sample = {}
            
            # Sample in topological order
            for node_name in self.topology:
                node = self.nodes[node_name]
                
                # If evidence is set, use it
                if node.evidence:
                    sample[node_name] = node.evidence
                    continue
                
                # Get parent states
                parent_states = {p: sample[p] for p in node.parents if p in sample}
                
                # Sample from distribution
                if not node.parents:
                    # Root node - sample from prior
                    probs = node.cpt
                else:
                    # Conditional node
                    parent_key = tuple(parent_states.get(p, None) for p in node.parents)
                    probs = node.cpt.get(parent_key, {})
                
                # Sample state
                states = list(probs.keys())
                probabilities = list(probs.values())
                
                if sum(probabilities) > 0:
                    probabilities = [p / sum(probabilities) for p in probabilities]
                    sample[node_name] = np.random.choice(states, p=probabilities)
                else:
                    sample[node_name] = states[0] if states else None
            
            samples.append(sample)
        
        return samples
    
    def explain_prediction(self, evidence: Dict[str, str]) -> Dict:
        """
        Provide explanation for fraud prediction
        
        Args:
            evidence: Dictionary of observed node states
            
        Returns:
            Explanation dictionary with contributing factors
        """
        fraud_probs = self.predict(evidence)
        
        # Calculate contribution of each factor
        contributions = {}
        
        for node_name in evidence.keys():
            # Calculate fraud probability without this evidence
            temp_evidence = evidence.copy()
            del temp_evidence[node_name]
            
            if temp_evidence:
                baseline_probs = self.predict(temp_evidence)
                contribution = fraud_probs['yes'] - baseline_probs['yes']
                contributions[node_name] = contribution
        
        return {
            'fraud_probability': fraud_probs['yes'],
            'fraud_class': 'yes' if fraud_probs['yes'] > 0.5 else 'no',
            'confidence': max(fraud_probs.values()),
            'contributing_factors': contributions,
            'evidence': evidence
        }
    
    def save_model(self, filepath: str):
        """Save the Bayesian Network to file"""
        model_data = {
            'topology': self.topology,
            'nodes': {
                name: {
                    'states': node.states,
                    'parents': node.parents,
                    'cpt': node.cpt
                }
                for name, node in self.nodes.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str):
        """Load the Bayesian Network from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.nodes = {}
        self.topology = model_data['topology']
        
        for name, node_data in model_data['nodes'].items():
            self.add_node(name, node_data['states'], node_data['parents'])
            self.nodes[name].set_cpt(node_data['cpt'])


def train_bayesian_network(training_data: pd.DataFrame) -> BayesianFraudDetector:
    """
    Train Bayesian Network from data
    
    Args:
        training_data: DataFrame with transaction features and fraud labels
        
    Returns:
        Trained BayesianFraudDetector
    """
    detector = BayesianFraudDetector()
    
    # In production, learn CPTs from data using maximum likelihood estimation
    # For now, using expert-defined probabilities
    
    return detector


if __name__ == '__main__':
    # Example usage
    detector = BayesianFraudDetector()
    
    # Test case 1: Low risk transaction
    evidence1 = {
        'transaction_amount': 'low',
        'transaction_frequency': 'normal',
        'location_change': 'no',
        'device_change': 'no',
        'time_of_day': 'afternoon',
        'merchant_category': 'low_risk',
        'user_history': 'trusted'
    }
    
    result1 = detector.explain_prediction(evidence1)
    print("Test Case 1 - Low Risk:")
    print(f"Fraud Probability: {result1['fraud_probability']:.4f}")
    print(f"Classification: {result1['fraud_class']}")
    print()
    
    # Test case 2: High risk transaction
    evidence2 = {
        'transaction_amount': 'very_high',
        'transaction_frequency': 'very_high',
        'location_change': 'international',
        'device_change': 'yes',
        'time_of_day': 'night',
        'merchant_category': 'high_risk',
        'user_history': 'new'
    }
    
    result2 = detector.explain_prediction(evidence2)
    print("Test Case 2 - High Risk:")
    print(f"Fraud Probability: {result2['fraud_probability']:.4f}")
    print(f"Classification: {result2['fraud_class']}")
    print(f"Contributing Factors: {result2['contributing_factors']}")
