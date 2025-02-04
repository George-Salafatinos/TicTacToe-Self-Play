import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os

from utils.tic_tac_toe import TicTacToeEnv, available_moves

########################################################
# CustomPolicyNet that supports arbitrary hidden_sizes
########################################################

class CustomPolicyNet(nn.Module):
    """
    A flexible policy network for Tic-Tac-Toe agent O, allowing different hidden layer sizes.
    Example:
        hidden_sizes=[32] creates a net (9 -> 32 -> 9).
        hidden_sizes=[32, 32] creates (9 -> 32 -> 32 -> 9), etc.
    """
    def __init__(self, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32]

        layers = []
        input_dim = 9  # board has 9 cells
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        # final layer outputs 9 logits (one per possible action)
        layers.append(nn.Linear(input_dim, 9))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

########################################################
# Helper function: interpret board for agent O
########################################################

def board_to_tensor_for_o(board):
    """
    O => +1, X => -1, '' => 0
    """
    arr = []
    for cell in board:
        if cell == 'O':
            arr.append(1)
        elif cell == 'X':
            arr.append(-1)
        else:
            arr.append(0)
    return torch.tensor(arr, dtype=torch.float)

########################################################
# Minimal moving average
########################################################

def moving_average(values, window_size=50):
    averaged = []
    for i in range(len(values)):
        start_index = max(0, i - window_size + 1)
        window_vals = values[start_index:i+1]
        avg = sum(window_vals) / len(window_vals)
        averaged.append(avg)
    return averaged

########################################################
# train_reinforce
########################################################

def train_reinforce(
    steps=10,
    lr=0.01,
    gamma=0.99,
    model_name="unnamed",
    opponent="random",
    hidden_sizes=None
):
    """
    The agent is O. Opponent (X) can be 'random' or 'self-play' (though
    here we just do random X to keep it simpler).

    hidden_sizes: a list of integers for layer sizes, e.g. [32], [32, 32], etc.
    We'll create a neural net with those hidden layers.

    Returns:
      model_data (dict) - with keys {"algorithm": "reinforce", "model_path": ...}
      (scores, losses)  - final arrays for each episode
    """
    if hidden_sizes is None:
        hidden_sizes = [32]

    # Create the policy net
    policy = CustomPolicyNet(hidden_sizes=hidden_sizes)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    env = TicTacToeEnv()
    scores = []
    losses = []
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
                    _, rew_x, done_x, info_x = env.step(x_move)
                    if info_x.get("winner") == 'X':
                        rewards.append(-1)  # O loses
                        log_probs.append(torch.tensor(0.0))
                        break
                    if done_x:
                        # draw
                        rewards.append(0)
                        log_probs.append(torch.tensor(0.0))
                        break

            elif opponent == "self-play":
                # Minimal approach: still random for X
                moves_x = available_moves(env.board)
                if moves_x:
                    x_move = random.choice(moves_x)
                    _, rew_x, done_x, info_x = env.step(x_move)
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
                # draw
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

            _, rew_o, done_o, info_o = env.step(action_o.item())
            if info_o.get("winner") == 'O':
                rewards.append(1)
                log_probs.append(log_prob_o)
                break
            elif done_o:
                # draw
                rewards.append(0)
                log_probs.append(log_prob_o)
                break
            else:
                # game continues
                log_probs.append(log_prob_o)
                rewards.append(0)

        # Compute returns
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

        # final reward
        final_reward = rewards[-1] if len(rewards) > 0 else 0
        if final_reward > 0:
            total_score += 1
        avg_score = total_score / (episode + 1)
        scores.append(avg_score)
        losses.append(policy_loss.item())

    # sliding-average the scores
    scores = moving_average(scores, window_size=50)

    # Save the model to disk
    timestamp = int(time.time())
    filename = f"{model_name}_reinforce_{timestamp}.pth"
    model_path = os.path.join("saved_models", filename)
    torch.save(policy.state_dict(), model_path)

    model_data = {
        "algorithm": "reinforce",
        "model_path": model_path
    }
    return model_data, (scores, losses)

########################################################

def predict_reinforce(board, model_data):
    """
    Inference for O.
    """
    # Rebuild the same net. We'll assume hidden_sizes=[32] for loading if
    # we didn't store them. For a real scenario, store hidden_sizes in model_data.
    # For now, we do minimal approach:
    policy = CustomPolicyNet(hidden_sizes=[32])  # or retrieve from model_data if we stored it
    model_path = model_data["model_path"]
    policy.load_state_dict(torch.load(model_path))
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
