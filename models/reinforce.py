import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os
from utils.tic_tac_toe import TicTacToeEnv, available_moves

class SimplePolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 32)
        self.fc2 = nn.Linear(32, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

def board_to_tensor_for_o(board):
    arr = []
    for cell in board:
        if cell == 'O':
            arr.append(1)
        elif cell == 'X':
            arr.append(-1)
        else:
            arr.append(0)
    return torch.tensor(arr, dtype=torch.float)

def moving_average(values, window_size=50):
    averaged = []
    for i in range(len(values)):
        start_index = max(0, i - window_size + 1)
        window_vals = values[start_index:i+1]
        avg = sum(window_vals) / len(window_vals)
        averaged.append(avg)
    return averaged

def train_reinforce(steps=10, lr=0.01, gamma=0.99, model_name="unnamed", opponent="random"):
    policy = SimplePolicyNet()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scores = []
    losses = []
    env = TicTacToeEnv()
    total_score = 0.0

    for episode in range(steps):
        log_probs = []
        rewards = []
        env.reset()
        done = False

        while not done:
            # X moves first
            if opponent == "random":
                moves_x = available_moves(env.board)
                if moves_x:
                    x_move = random.choice(moves_x)
                    next_state, rew_x, done_x, info_x = env.step(x_move)
                    if info_x.get("winner") == 'X':
                        rewards.append(-1)
                        log_probs.append(torch.tensor(0.0))
                        break
                    if done_x:
                        rewards.append(0)
                        log_probs.append(torch.tensor(0.0))
                        break
            elif opponent == "self-play":
                moves_x = available_moves(env.board)
                if moves_x:
                    x_move = random.choice(moves_x)
                    next_state, rew_x, done_x, info_x = env.step(x_move)
                    if info_x.get("winner") == 'X':
                        rewards.append(-1)
                        log_probs.append(torch.tensor(0.0))
                        break
                    if done_x:
                        rewards.append(0)
                        log_probs.append(torch.tensor(0.0))
                        break

            if env.is_done():
                break

            # O's turn
            state_o = board_to_tensor_for_o(env.board).unsqueeze(0)
            logits_o = policy(state_o)
            moves_o = available_moves(env.board)
            if not moves_o:
                rewards.append(0)
                log_probs.append(torch.tensor(0.0))
                break

            mask_o = torch.zeros(9)
            for mo in moves_o:
                mask_o[mo] = 1
            masked_logits_o = logits_o + (mask_o.unsqueeze(0) - 1) * 1e6
            dist_o = torch.distributions.Categorical(logits=masked_logits_o[0])
            action_o = dist_o.sample()
            log_prob_o = dist_o.log_prob(action_o)
            next_state, rew_o, done_o, info_o = env.step(action_o.item())

            if info_o.get("winner") == 'O':
                rewards.append(1)
                log_probs.append(log_prob_o)
                break
            elif done_o:
                rewards.append(0)
                log_probs.append(log_prob_o)
                break
            else:
                log_probs.append(log_prob_o)
                rewards.append(0)

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = torch.stack([-lp * R for lp, R in zip(log_probs, returns)]).sum()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        final_r = rewards[-1] if len(rewards) > 0 else 0
        if final_r > 0:
            total_score += 1
        avg_score = total_score / (episode + 1)
        scores.append(avg_score)

        losses.append(policy_loss.item())

    scores = moving_average(scores, window_size=50)
    timestamp = int(time.time())
    model_path = os.path.join("saved_models", f"{model_name}_reinforce_{timestamp}.pth")
    torch.save(policy.state_dict(), model_path)
    model_data = {
        "algorithm": "reinforce",
        "model_path": model_path
    }
    # Return (scores, losses)
    return model_data, (scores, losses)

def predict_reinforce(board, model_data):
    policy = SimplePolicyNet()
    policy.load_state_dict(torch.load(model_data["model_path"]))
    policy.eval()

    state_o = board_to_tensor_for_o(board).unsqueeze(0)
    logits_o = policy(state_o)
    moves_o = available_moves(board)
    if not moves_o:
        return None

    mask_o = torch.zeros(9)
    for mo in moves_o:
        mask_o[mo] = 1
    masked_logits_o = logits_o + (mask_o.unsqueeze(0) - 1) * 1e6
    dist_o = torch.distributions.Categorical(logits=masked_logits_o[0])
    action_o = dist_o.sample()
    return action_o.item()
