import random
from utils.tic_tac_toe import TicTacToeEnv, available_moves

def train_random_search(steps=10):
    env = TicTacToeEnv()
    avg_scores = []
    best_score = -999
    best_params = {"random_seed": 42}
    total_score = 0.0

    for i in range(steps):
        candidate_seed = random.randint(0, 100000)
        random.seed(candidate_seed)
        env.reset()
        done = False
        while not done:
            moves = available_moves(env.board)
            if not moves:
                break
            action = random.choice(moves)
            next_state, reward, done, info = env.step(action)
        winner = info.get("winner", None)
        score = 1 if winner == 'X' else 0
        total_score += score
        current_avg = total_score / (i + 1)
        avg_scores.append(current_avg)
        if score > best_score:
            best_score = score
            best_params = {"random_seed": candidate_seed}

    trained_model = {
        "algorithm": "random-search",
        "best_params": best_params,
        "best_score": best_score
    }
    return trained_model, avg_scores

def predict_random_search(board, model_data):
    best_seed = model_data.get("best_params", {}).get("random_seed", 42)
    random.seed(best_seed)
    moves = available_moves(board)
    if not moves:
        return None
    return random.choice(moves)
