from models.random_search import train_random_search

def train_selected_model(algorithm_name, hyperparams):
    """
    Dispatch logic to call the correct training function based on algorithm_name.
    For now, only random-search is implemented.
    
    hyperparams is expected to be a dict, e.g. {"steps": 10}.
    """
    if algorithm_name == "random-search":
        steps = hyperparams.get("steps", 10)
        model_info = train_random_search(steps=steps)
        return {
            "algorithm": algorithm_name,
            "hyperparams": hyperparams,
            "model_info": model_info
        }
    else:
        # Future expansions: REINFORCE, PPO, etc.
        return {
            "algorithm": algorithm_name,
            "error": "Algorithm not implemented yet."
        }
