import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque
import itertools

class Factor:
    """Represents a factor in the Bayesian Network"""
    
    def __init__(self, variables: List[str], values: Dict[Tuple, float]):
        self.variables = variables
        self.values = values
        
    def multiply(self, other: 'Factor') -> 'Factor':
        """Multiply two factors"""
        # Get union of variables
        new_vars = list(set(self.variables + other.variables))
        new_values = {}
        
        # Generate all combinations
        for assignment in self._generate_assignments(new_vars):
            # Get values from both factors
            self_key = tuple(assignment[v] for v in self.variables)
            other_key = tuple(assignment[v] for v in other.variables)
            
            self_val = self.values.get(self_key, 0)
            other_val = other.values.get(other_key, 0)
            
            new_key = tuple(assignment[v] for v in new_vars)
            new_values[new_key] = self_val * other_val
        
        return Factor(new_vars, new_values)
    
    def marginalize(self, variable: str) -> 'Factor':
        """Marginalize out a variable"""
        if variable not in self.variables:
            return self
        
        new_vars = [v for v in self.variables if v != variable]
        new_values = defaultdict(float)
        
        var_index = self.variables.index(variable)
        
        for key, value in self.values.items():
            new_key = tuple(k for i, k in enumerate(key) if i != var_index)
            new_values[new_key] += value
        
        return Factor(new_vars, dict(new_values))
    
    def reduce(self, variable: str, value: str) -> 'Factor':
        """Reduce factor by setting variable to value"""
        if variable not in self.variables:
            return self
        
        new_vars = [v for v in self.variables if v != variable]
        new_values = {}
        
        var_index = self.variables.index(variable)
        
        for key, prob in self.values.items():
            if key[var_index] == value:
                new_key = tuple(k for i, k in enumerate(key) if i != var_index)
                new_values[new_key] = prob
        
        return Factor(new_vars, new_values)
    
    def normalize(self) -> 'Factor':
        """Normalize factor to sum to 1"""
        total = sum(self.values.values())
        if total == 0:
            return self
        
        new_values = {k: v / total for k, v in self.values.items()}
        return Factor(self.variables, new_values)
    
    def _generate_assignments(self, variables: List[str]) -> List[Dict[str, str]]:
        """Generate all possible assignments for variables"""
        # Simplified - in production, get actual domains from network
        domains = {v: ['state1', 'state2'] for v in variables}
        
        assignments = []
        for values in itertools.product(*[domains[v] for v in variables]):
            assignment = dict(zip(variables, values))
            assignments.append(assignment)
        
        return assignments


class InferenceEngine:
    """Advanced inference engine for Bayesian Networks"""
    
    def __init__(self, network):
        self.network = network
        self.factors = {}
        self.elimination_order = []
        
    def variable_elimination(self, query_vars: List[str], 
                           evidence: Dict[str, str]) -> Dict[Tuple, float]:
        """
        Perform variable elimination inference
        
        Args:
            query_vars: Variables to query
            evidence: Observed evidence
            
        Returns:
            Joint probability distribution over query variables
        """
        # Initialize factors from CPTs
        factors = self._initialize_factors()
        
        # Reduce factors based on evidence
        factors = [f.reduce(var, val) for f in factors 
                  for var, val in evidence.items()]
        
        # Determine elimination order
        elimination_vars = [v for v in self.network.topology 
                          if v not in query_vars and v not in evidence]
        
        # Eliminate variables
        for var in elimination_vars:
            factors = self._eliminate_variable(var, factors)
        
        # Multiply remaining factors
        result = factors[0]
        for f in factors[1:]:
            result = result.multiply(f)
        
        # Normalize
        result = result.normalize()
        
        return result.values
    
    def belief_propagation(self, query_var: str, 
                          evidence: Dict[str, str],
                          max_iterations: int = 100) -> Dict[str, float]:
        """
        Perform belief propagation (sum-product algorithm)
        
        Args:
            query_var: Variable to query
            evidence: Observed evidence
            max_iterations: Maximum iterations
            
        Returns:
            Probability distribution over query variable
        """
        # Initialize messages
        messages = self._initialize_messages()
        
        # Set evidence
        for var, val in evidence.items():
            if var in self.network.nodes:
                self.network.nodes[var].set_evidence(val)
        
        # Iterate message passing
        for iteration in range(max_iterations):
            old_messages = messages.copy()
            
            # Update messages
            for node_name in self.network.topology:
                node = self.network.nodes[node_name]
                
                # Send messages to children
                for child in self._get_children(node_name):
                    messages[(node_name, child)] = self._compute_message(
                        node_name, child, messages
                    )
            
            # Check convergence
            if self._has_converged(messages, old_messages):
                break
        
        # Compute marginal for query variable
        marginal = self._compute_marginal(query_var, messages)
        
        # Clear evidence
        for node in self.network.nodes.values():
            node.clear_evidence()
        
        return marginal
    
    def map_inference(self, evidence: Dict[str, str]) -> Dict[str, str]:
        """
        Maximum a posteriori (MAP) inference
        
        Args:
            evidence: Observed evidence
            
        Returns:
            Most probable assignment to all variables
        """
        # Use max-product algorithm
        factors = self._initialize_factors()
        
        # Reduce factors based on evidence
        for var, val in evidence.items():
            factors = [f.reduce(var, val) for f in factors]
        
        # Find MAP assignment
        map_assignment = {}
        
        for node_name in reversed(self.network.topology):
            if node_name in evidence:
                map_assignment[node_name] = evidence[node_name]
                continue
            
            # Find value that maximizes probability
            node = self.network.nodes[node_name]
            max_prob = -1
            max_state = None
            
            for state in node.states:
                # Compute probability for this state
                temp_evidence = {**evidence, **map_assignment, node_name: state}
                prob = self._compute_joint_probability(temp_evidence)
                
                if prob > max_prob:
                    max_prob = prob
                    max_state = state
            
            map_assignment[node_name] = max_state
        
        return map_assignment
    
    def sensitivity_analysis(self, query_var: str, 
                            evidence: Dict[str, str],
                            parameter_node: str,
                            parameter_state: str) -> Dict[str, List[float]]:
        """
        Perform sensitivity analysis on network parameters
        
        Args:
            query_var: Variable to analyze
            evidence: Observed evidence
            parameter_node: Node whose parameter to vary
            parameter_state: State whose probability to vary
            
        Returns:
            Sensitivity curves
        """
        results = {state: [] for state in self.network.nodes[query_var].states}
        
        # Vary parameter from 0 to 1
        original_cpt = self.network.nodes[parameter_node].cpt.copy()
        
        for p in np.linspace(0, 1, 20):
            # Modify CPT
            self._modify_cpt(parameter_node, parameter_state, p)
            
            # Compute query
            probs = self.variable_elimination([query_var], evidence)
            
            # Store results
            for state in results.keys():
                results[state].append(probs.get((state,), 0))
        
        # Restore original CPT
        self.network.nodes[parameter_node].cpt = original_cpt
        
        return results
    
    def _initialize_factors(self) -> List[Factor]:
        """Initialize factors from CPTs"""
        factors = []
        
        for node_name, node in self.network.nodes.items():
            variables = [node_name] + node.parents
            values = {}
            
            # Convert CPT to factor
            if not node.parents:
                # Root node
                for state in node.states:
                    values[(state,)] = node.cpt.get(state, 0)
            else:
                # Conditional node
                for parent_assignment in self._get_parent_assignments(node.parents):
                    parent_key = tuple(parent_assignment[p] for p in node.parents)
                    
                    for state in node.states:
                        key = (state,) + parent_key
                        prob = node.cpt.get(parent_key, {}).get(state, 0)
                        values[key] = prob
            
            factors.append(Factor(variables, values))
        
        return factors
    
    def _eliminate_variable(self, variable: str, 
                           factors: List[Factor]) -> List[Factor]:
        """Eliminate a variable from factors"""
        # Find factors containing variable
        relevant_factors = [f for f in factors if variable in f.variables]
        other_factors = [f for f in factors if variable not in f.variables]
        
        if not relevant_factors:
            return factors
        
        # Multiply relevant factors
        product = relevant_factors[0]
        for f in relevant_factors[1:]:
            product = product.multiply(f)
        
        # Marginalize out variable
        marginalized = product.marginalize(variable)
        
        return other_factors + [marginalized]
    
    def _initialize_messages(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Initialize messages for belief propagation"""
        messages = {}
        
        for node_name in self.network.topology:
            node = self.network.nodes[node_name]
            
            # Initialize messages to children
            for child in self._get_children(node_name):
                messages[(node_name, child)] = {
                    state: 1.0 / len(node.states) for state in node.states
                }
        
        return messages
    
    def _compute_message(self, from_node: str, to_node: str,
                        messages: Dict) -> Dict[str, float]:
        """Compute message from one node to another"""
        node = self.network.nodes[from_node]
        message = {}
        
        for state in node.states:
            # If evidence is set, message is deterministic
            if node.evidence:
                message[state] = 1.0 if state == node.evidence else 0.0
            else:
                # Compute message based on CPT and incoming messages
                message[state] = self._compute_message_value(
                    from_node, to_node, state, messages
                )
        
        # Normalize
        total = sum(message.values())
        if total > 0:
            message = {k: v / total for k, v in message.items()}
        
        return message
    
    def _compute_message_value(self, from_node: str, to_node: str,
                               state: str, messages: Dict) -> float:
        """Compute message value for a specific state"""
        # Simplified computation
        return 1.0
    
    def _compute_marginal(self, variable: str, 
                         messages: Dict) -> Dict[str, float]:
        """Compute marginal probability for a variable"""
        node = self.network.nodes[variable]
        marginal = {}
        
        for state in node.states:
            # Product of incoming messages
            prob = 1.0
            
            for parent in node.parents:
                if (parent, variable) in messages:
                    prob *= messages[(parent, variable)].get(state, 1.0)
            
            marginal[state] = prob
        
        # Normalize
        total = sum(marginal.values())
        if total > 0:
            marginal = {k: v / total for k, v in marginal.items()}
        
        return marginal
    
    def _has_converged(self, messages: Dict, old_messages: Dict,
                      threshold: float = 1e-6) -> bool:
        """Check if belief propagation has converged"""
        for key in messages:
            if key not in old_messages:
                return False
            
            for state in messages[key]:
                diff = abs(messages[key][state] - old_messages[key].get(state, 0))
                if diff > threshold:
                    return False
        
        return True
    
    def _get_children(self, node_name: str) -> List[str]:
        """Get children of a node"""
        children = []
        
        for name, node in self.network.nodes.items():
            if node_name in node.parents:
                children.append(name)
        
        return children
    
    def _get_parent_assignments(self, parents: List[str]) -> List[Dict[str, str]]:
        """Get all possible parent assignments"""
        if not parents:
            return [{}]
        
        assignments = []
        parent_states = [self.network.nodes[p].states for p in parents]
        
        for values in itertools.product(*parent_states):
            assignment = dict(zip(parents, values))
            assignments.append(assignment)
        
        return assignments
    
    def _compute_joint_probability(self, assignment: Dict[str, str]) -> float:
        """Compute joint probability of an assignment"""
        prob = 1.0
        
        for node_name in self.network.topology:
            node = self.network.nodes[node_name]
            state = assignment.get(node_name)
            
            if state is None:
                continue
            
            parent_states = {p: assignment.get(p) for p in node.parents}
            node_prob = node.get_probability(state, parent_states)
            prob *= node_prob
        
        return prob
    
    def _modify_cpt(self, node_name: str, state: str, probability: float):
        """Modify CPT for sensitivity analysis"""
        node = self.network.nodes[node_name]
        
        if not node.parents:
            # Adjust probabilities to maintain normalization
            other_states = [s for s in node.states if s != state]
            remaining_prob = 1.0 - probability
            
            node.cpt[state] = probability
            for s in other_states:
                node.cpt[s] = remaining_prob / len(other_states)
