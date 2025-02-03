from models.random_search import train_random_search
from utils.plot import plot_training_curve

def train_selected_model(algorithm_name, hyperparams):
    """
    Dispatch logic to call the correct training function based on algorithm_name.
    For now, only random-search is implemented with metrics for plotting.
    """
    if algorithm_name == "random-search":
        steps = hyperparams.get("steps", 10)
        trained_model, avg_scores = train_random_search(steps=steps)

        # Generate a chart of the average scores
        chart_b64 = plot_training_curve(avg_scores, title="Cumulative Avg Score per Iteration")

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
