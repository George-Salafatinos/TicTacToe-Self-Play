import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os
from utils.tic_tac_toe import TicTacToeEnv, available_moves, check_winner

class SimplePolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 32)
        self.fc2 = nn.Linear(32, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

def board_to_tensor(board, player='X'):
    mapping = {'X': 1, 'O': -1, '': 0}
    arr = [mapping[cell] for cell in board]
    return torch.tensor(arr, dtype=torch.float)

def train_reinforce(steps=10, lr=0.01, gamma=0.85):
    policy = SimplePolicyNet()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scores = []
    env = TicTacToeEnv()
    total_score = 0.0

    for episode in range(steps):
        log_probs = []
        rewards = []
        env.reset()
        done = False

        while not done:
            state_tensor = board_to_tensor(env.board, player='X').unsqueeze(0)
            logits = policy(state_tensor)
            moves = available_moves(env.board)
            mask = torch.zeros(9)
            for m in moves:
                mask[m] = 1
            masked_logits = logits + (mask.unsqueeze(0) - 1) * 1e6
            action_dist = torch.distributions.Categorical(logits=masked_logits[0])
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            next_state, reward, done, info = env.step(action.item())
            if not done:
                moves_o = available_moves(env.board)
                if moves_o:
                    a_o = random.choice(moves_o)
                    next_state, reward_o, done, info_o = env.step(a_o)
                    if info_o.get("winner") == 'O':
                        reward = -1
                        done = True
            log_probs.append(log_prob)
            rewards.append(reward)

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        policy_loss = []
        for lp, R in zip(log_probs, returns):
            policy_loss.append(-lp * R)
        policy_loss = torch.stack(policy_loss).sum()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        final_reward = rewards[-1] if len(rewards) > 0 else 0
        if final_reward > 0:
            total_score += 1
        current_avg = total_score / (episode + 1)
        scores.append(current_avg)

    model_filename = "reinforce_" + str(int(time.time())) + ".pth"
    model_path = os.path.join("saved_models", model_filename)
    torch.save(policy.state_dict(), model_path)

    model_data = {
        "algorithm": "reinforce",
        "model_path": model_path
    }
    return model_data, scores

def predict_reinforce(board, model_data):
    from utils.tic_tac_toe import available_moves
    policy = SimplePolicyNet()
    model_path = model_data["model_path"]
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    state_tensor = board_to_tensor(board, player='X').unsqueeze(0)
    logits = policy(state_tensor)
    moves = available_moves(board)
    if not moves:
        return None
    mask = torch.zeros(9)
    for m in moves:
        mask[m] = 1
    masked_logits = logits + (mask.unsqueeze(0) - 1) * 1e6
    action_dist = torch.distributions.Categorical(logits=masked_logits[0])
    action = action_dist.sample()
    return action.item()
