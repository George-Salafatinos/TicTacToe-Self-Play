from utils.plot import plot_training_curve
from models.random_search import train_random_search, predict_random_search
from models.reinforce import train_reinforce, predict_reinforce

ACTIVE_MODELS = {}

def train_selected_model(algorithm_name, hyperparams):
    steps = int(hyperparams.get("steps", 10))
    model_name = str(hyperparams.get("model_name", "unnamed"))
    opponent = hyperparams.get("opponent", "random")
    lr = float(hyperparams.get("lr", 0.01))
    gamma = float(hyperparams.get("gamma", 0.99))

    if algorithm_name == "random-search":
        trained_model, avg_scores = train_random_search(steps=steps)
        chart_b64 = plot_training_curve(
            avg_scores,
            title=f"Random Search (Agent=O?) - {model_name} vs {opponent}"
        )
        ACTIVE_MODELS[algorithm_name] = trained_model
        return {
            "algorithm": algorithm_name,
            "hyperparams": hyperparams,
            "model_info": {
                "algorithm": "random-search",
                "best_score": trained_model["best_score"]
            },
            "chart_b64": chart_b64
        }

    elif algorithm_name == "reinforce":
        trained_model, scores = train_reinforce(
            steps=steps,
            lr=lr,
            gamma=gamma,
            model_name=model_name,
            opponent=opponent
        )
        chart_b64 = plot_training_curve(
            scores,
            title=f"REINFORCE (Agent=O) - {model_name} vs {opponent}"
        )
        ACTIVE_MODELS[algorithm_name] = trained_model
        return {
            "algorithm": algorithm_name,
            "hyperparams": hyperparams,
            "model_info": {
                "algorithm": "reinforce",
                "saved_path": trained_model["model_path"]
            },
            "chart_b64": chart_b64
        }

    else:
        return {
            "algorithm": algorithm_name,
            "error": "Algorithm not implemented yet."
        }

def predict_move(algorithm_name, board_state):
    model_data = ACTIVE_MODELS.get(algorithm_name)
    if not model_data:
        return None
    if algorithm_name == "random-search":
        return predict_random_search(board_state, model_data)
    if algorithm_name == "reinforce":
        return predict_reinforce(board_state, model_data)
    return None
