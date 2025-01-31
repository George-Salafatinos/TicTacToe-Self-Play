from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.learning_rate = learning_rate
        
    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """
        Update policy based on experience
        Returns: loss value for logging
        """
        pass
    
    @abstractmethod
    def get_action(self, state, valid_moves, training=True):
        """
        Get action from policy, with potential exploration
        Returns: chosen action
        """
        pass
    
    def get_config(self):
        """Returns algorithm configuration"""
        return {
            "learning_rate": self.learning_rate,
            "model_config": self.model.get_config()
        }