import gym
from gym import spaces
import numpy as np
from typing import Tuple, Dict

class FraudDetectionEnvironment(gym.Env):
    def __init__(self, transaction_data: np.ndarray, labels: np.ndarray):
        super(FraudDetectionEnvironment, self).__init__()
        
        self.transaction_data = transaction_data
        self.labels = labels
        self.current_step = 0
        self.max_steps = len(transaction_data)
        
        # Define action space: 0 = approve, 1 = reject, 2 = review
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(transaction_data.shape[1],),
            dtype=np.float32
        )
        
        # Reward parameters
        self.true_positive_reward = 100
        self.true_negative_reward = 1
        self.false_positive_cost = -10
        self.false_negative_cost = -50
        self.review_cost = -2
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        return self.transaction_data[self.current_step]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state"""
        current_transaction = self.transaction_data[self.current_step]
        true_label = self.labels[self.current_step]
        
        # Calculate reward
        reward = self._calculate_reward(action, true_label)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Get next observation
        next_observation = (
            self.transaction_data[self.current_step]
            if not done
            else np.zeros_like(current_transaction)
        )
        
        info = {
            'true_label': true_label,
            'action': action,
            'step': self.current_step
        }
        
        return next_observation, reward, done, info
    
    def _calculate_reward(self, action: int, true_label: int) -> float:
        """Calculate reward based on action and true label"""
        if action == 2:  # Review
            return self.review_cost
        
        if true_label == 1:  # Fraud
            if action == 1:  # Reject
                return self.true_positive_reward
            else:  # Approve
                return self.false_negative_cost
        else:  # Legitimate
            if action == 0:  # Approve
                return self.true_negative_reward
            else:  # Reject
                return self.false_positive_cost
    
    def render(self, mode='human'):
        """Render environment state"""
        print(f"Step: {self.current_step}/{self.max_steps}")
