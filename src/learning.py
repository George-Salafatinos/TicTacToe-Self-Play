# learning.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tictactoe_env import TicTacToeEnv

class TicTacToeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Network layers
        # Input: 10 (9 board positions + current player)
        # Output: 9 (probability for each position)

        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass logic
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class TicTacToeAgent:
    def __init__(self):
        self.model = TicTacToeNet()
        self.optimizer = optim.Adam(self.model.parameters())

    def get_action(self, state, valid_moves):
        # Use model to select action

        # Convert state to tensor and add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        #Get model predictions (using with torch.no_grad() since we're not training)
        with torch.no_grad():
            logits = self.model(state_tensor)
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1).squeeze()

        probs = probs.numpy()
        
        # Mask invalid moves
        mask = np.zeros(9)
        mask[valid_moves] = 1
        probs = probs * mask

        # Normalize probabilities
        if np.sum(probs)>0:
            probs = probs / np.sum(probs)
        else:
            probs = mask / mask.sum()

        # Choose action
        action = np.random.choice(9, p=probs)

        # Return action
        return action


    def update_policy(self, state, action, reward, entropy_coef=0.01):
        """
        state: the game state
        action: the action that was taken
        reward: +1 if won, -1 if lost
        entropy_coef: coefficient for entropy regularization
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get model predictions
        logits = self.model(state_tensor)
        probs = torch.softmax(logits, dim=1).squeeze()
        
        # Get log probability of the action that was taken
        log_prob = torch.log(probs[action])
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        
        # Calculate loss with entropy regularization
        # We subtract entropy (negative sign) because we're minimizing loss 
        # but want to maximize entropy
        loss = -log_prob * reward - entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_weights(self, filepath):
        # Save model weights
        torch.save(self.model.state_dict(), filepath)
        
    def load_weights(self, filepath):
        # Load model weights
        self.model.load_state_dict(torch.load(filepath))

def play_against_self(agent):
    env = TicTacToeEnv()
    experiences_player = []  # List to store (state, action, next_state, done)
    experiences_opponent = []
    state = env.reset()
    agent_player = np.random.choice([-1,1])  # Randomly decide if agent is X or O
    
    while not env.done:
        current_player = env.current_player
        valid_moves = env.get_valid_moves()
        action = agent.get_action(state, valid_moves)
        next_state, reward, done = env.make_move(action)
        if current_player == agent_player:
            experiences_player.append([state, action, next_state, done])
        state = next_state

    return experiences_player, env.winner == agent_player


def play_against_random(agent):
    env = TicTacToeEnv()
    experiences = []  # Only store agent's moves
    state = env.reset()
    agent_player = np.random.choice([-1,1])  # Randomly decide if agent is X or O
    
    while not env.done:
        valid_moves = env.get_valid_moves()
        
        if env.current_player == agent_player:
            # Agent's turn
            action= agent.get_action(state, valid_moves)
        else:
            # Random opponent's turn
            action = np.random.choice(valid_moves)
            
        next_state, reward, done = env.make_move(action)
        
        # Only store agent's moves
        if env.current_player == agent_player:
            experiences.append([state, action, next_state, done])
            
        state = next_state

    return [experiences], env.winner == agent_player

def play_against_opponent_agent(agent1, agent2):
    env = TicTacToeEnv()
    experiences_agent1 = []  # Store agent1's moves
    experiences_agent2 = []  # Store agent2's moves
    state = env.reset()
    agent1_player = np.random.choice([-1,1])  # Randomly decide if agent1 is X or O
    
    while not env.done:
        valid_moves = env.get_valid_moves()
        
        if env.current_player == agent1_player:
            # Agent1's turn
            action = agent1.get_action(state, valid_moves)
            next_state, reward, done = env.make_move(action)
            experiences_agent1.append([state, action, next_state, done])
        else:
            # Agent2's turn
            action = agent2.get_action(state, valid_moves)
            next_state, reward, done = env.make_move(action)
            experiences_agent2.append([state, action, next_state, done])
            
        state = next_state

    return [experiences_agent1, experiences_agent2], env.winner == agent1_player
