import numpy as np
from .base_model import BaseModel

class SimpleModel(BaseModel):
    def __init__(self, input_dim=9, hidden_dim=64):
        super().__init__(input_dim, hidden_dim)
        # Initialize weights with Xavier/Glorot initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 9) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(9)
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
        
    def predict_action(self, state, valid_moves):
        # Forward pass
        h = self.relu(state @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        
        # Mask invalid moves with large negative number
        masked_logits = logits.copy()
        invalid_moves = [i for i in range(9) if i not in valid_moves]
        masked_logits[invalid_moves] = -1e9
        
        # Get probabilities and sample
        probs = self.softmax(masked_logits)
        return np.random.choice(9, p=probs)
    
    def get_model_parameters(self):
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }
    
    def set_model_parameters(self, parameters):
        self.W1 = parameters['W1']
        self.b1 = parameters['b1']
        self.W2 = parameters['W2']
        self.b2 = parameters['b2']