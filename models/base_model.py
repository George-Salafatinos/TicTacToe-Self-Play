from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    def __init__(self, input_dim=9, hidden_dim=64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
    @abstractmethod
    def predict_action(self, state, valid_moves):
        """
        Given a state and valid moves, predict next action
        Returns: action index (0-8)
        """
        pass
    
    @abstractmethod
    def get_model_parameters(self):
        """Returns model parameters for saving"""
        pass
    
    @abstractmethod
    def set_model_parameters(self, parameters):
        """Sets model parameters from loaded data"""
        pass
        
    def get_config(self):
        """Returns model configuration"""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim
        }