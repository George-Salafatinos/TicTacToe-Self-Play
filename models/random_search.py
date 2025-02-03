import random
from utils.tic_tac_toe import TicTacToeEnv, available_moves

def train_random_search(steps=10):
    """
    A toy 'random search' approach for demonstration.
    We'll accumulate a 'score' each iteration.
    Instead of returning raw 0/1 for each step,
    we'll keep a running average for a smoother plot.
    """
    env = TicTacToeEnv()
    
    avg_scores = []  # cumulative average score
    best_score = -999
    best_params = {"random_seed": 42}

    total_score = 0.0  # running sum of scores over iterations
    for i in range(steps):
        # Try a new random seed
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

        # Check winner
        winner = info.get("winner", None)
        # For demonstration: if 'X' won, we treat it as a 1, else 0.
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

    # Return the trained model and the average scores
    return trained_model, avg_scores
