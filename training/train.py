from models.random_search import train_random_search, predict_random_search
from utils.plot import plot_training_curve

# This dictionary will store the "latest trained model" for each algorithm.
ACTIVE_MODELS = {}

def train_selected_model(algorithm_name, hyperparams):
    if algorithm_name == "random-search":
        steps = hyperparams.get("steps", 10)
        trained_model, avg_scores = train_random_search(steps=steps)

        chart_b64 = plot_training_curve(avg_scores, title="Cumulative Avg Score per Iteration")

        # Store in global dict
        ACTIVE_MODELS[algorithm_name] = trained_model

        return {
            "algorithm": algorithm_name,
            "hyperparams": hyperparams,
            "model_info": trained_model,
            "chart_b64": chart_b64
        }
    else:
        return {
            "algorithm": algorithm_name,
            "error": "Algorithm not implemented yet."
        }

def predict_move(algorithm_name, board_state):
    """
    Retrieve the trained model from ACTIVE_MODELS,
    call the appropriate predict function, and return the chosen move.
    """
    model_data = ACTIVE_MODELS.get(algorithm_name)
    if not model_data:
        return None  # No model loaded/trained for that algorithm

    if algorithm_name == "random-search":
        return predict_random_search(board_state, model_data)
    else:
        return None
