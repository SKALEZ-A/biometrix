import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(256, input_dim=self.state_size, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        targets = rewards + self.gamma * (np.amax(self.target_model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        
        ind = np.array([i for i in range(batch_size)])
        targets_full[[ind], [actions]] = targets
        
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class FraudDetectionEnvironment:
    def __init__(self):
        self.state_size = 50
        self.action_size = 3
        self.current_state = None
        self.transaction_history = deque(maxlen=1000)
        
    def reset(self):
        self.current_state = np.zeros(self.state_size)
        return self.current_state
    
    def step(self, action, transaction_data):
        reward = self._calculate_reward(action, transaction_data)
        next_state = self._get_next_state(transaction_data)
        done = False
        
        self.transaction_history.append({
            'state': self.current_state,
            'action': action,
            'reward': reward,
            'transaction': transaction_data
        })
        
        self.current_state = next_state
        return next_state, reward, done
    
    def _calculate_reward(self, action, transaction_data):
        is_fraud = transaction_data.get('is_fraud', False)
        
        if action == 0:
            return 1 if not is_fraud else -10
        elif action == 1:
            return 0.5 if not is_fraud else -5
        elif action == 2:
            return -1 if not is_fraud else 10
        
        return 0
    
    def _get_next_state(self, transaction_data):
        features = np.array([
            transaction_data.get('amount', 0),
            transaction_data.get('velocity', 0),
            transaction_data.get('location_risk', 0),
            transaction_data.get('device_risk', 0),
            transaction_data.get('behavioral_score', 0)
        ])
        
        state = np.zeros(self.state_size)
        state[:len(features)] = features
        return state
