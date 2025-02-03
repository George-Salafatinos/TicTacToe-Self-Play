import random
from utils.tictactoe import TicTacToeEnv, available_moves

def train_random_search(steps=10):
    """
    A toy 'random search' approach for demonstration.
    We'll just simulate some games and pick random actions.
    This doesn't really "learn," but we'll pretend by returning pseudo-results.

    steps: Number of simulated iterations to 'search' for best strategy.
    Returns a dict representing the 'trained model'.
    """
    env = TicTacToeEnv()

    best_score = -999
    best_params = {"random_seed": 42}  # Fake param store

    for i in range(steps):
        # "Simulate" a new random seed each iteration
        candidate_seed = random.randint(0, 100000)
        random.seed(candidate_seed)

        # Just run one game randomly, track if we got a 'win'
        env.reset()
        done = False
        while not done:
            moves = available_moves(env.board)
            if not moves:
                break
            action = random.choice(moves)
            next_state, reward, done, info = env.step(action)

        # If X or O won with our random seed, treat that as a 'score'
        # This is purely for demonstration
        winner = info.get("winner", None)
        if winner == 'X':
            score = 1
        else:
            score = 0

        if score > best_score:
            best_score = score
            best_params = {"random_seed": candidate_seed}

    # Return something that represents our "best found params"
    trained_model = {
        "algorithm": "random-search",
        "best_params": best_params,
        "best_score": best_score
    }
    return trained_model
