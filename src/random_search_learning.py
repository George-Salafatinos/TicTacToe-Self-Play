import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tictactoe_env import TicTacToeEnv

class TicTacToeNetRS(nn.Module):
    def __init__(self):
        super().__init__()
        # Same network structure as original for fair comparison
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RandomSearchAgent:
    def __init__(self, step_size=0.1, momentum=2):
        self.model = TicTacToeNetRS()
        self.step_size = step_size
        self.momentum = momentum
        self.prev_direction = None
        
    def get_action(self, state, valid_moves):
        # Similar to original agent's get_action
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(state_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()
        
        probs = probs.numpy()
        mask = np.zeros(9)
        mask[valid_moves] = 1
        probs = probs * mask
        
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        else:
            probs = mask / mask.sum()
            
        action = np.random.choice(9, p=probs)
        return action

    def get_random_direction(self):
        direction = []
        with torch.no_grad():
            for param in self.model.parameters():
                random_direction = torch.randn_like(param.data)
                random_direction = random_direction / torch.norm(random_direction)
                direction.append(random_direction)
        return direction
        
    def apply_direction(self, direction, scale=1.0):
        with torch.no_grad():
            for param, dir_tensor in zip(self.model.parameters(), direction):
                param.data += scale * self.step_size * dir_tensor
    
    def update_from_game(self, won):
        current_direction = self.get_random_direction()
        
        if self.prev_direction is not None:
            # Add momentum from previous successful direction
            for curr, prev in zip(current_direction, self.prev_direction):
                curr.data += self.momentum * prev.data
                curr.data = curr.data / torch.norm(curr.data)
        
        # Apply direction based on game outcome
        scale = 1.0 if won else -1.0
        self.apply_direction(current_direction, scale)
        self.prev_direction = current_direction

    def save_weights(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        
    def load_weights(self, filepath):
        self.model.load_state_dict(torch.load(filepath))