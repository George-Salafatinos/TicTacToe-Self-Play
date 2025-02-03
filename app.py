from flask import Flask, render_template, request, jsonify
import os

from training.train import train_selected_model, predict_move

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train-model", methods=["POST"])
def train_model():
    data = request.json
    algorithm = data.get("algorithm", "")
    hyperparams = data.get("hyperparams", {})

    if not algorithm:
        return jsonify({"message": "No algorithm specified"}), 400

    result = train_selected_model(algorithm, hyperparams)
    chart_b64 = result.get("chart_b64", None)

    return jsonify({
        "message": "Training complete!",
        "details": result,
        "chart_b64": chart_b64
    })

@app.route("/select-model", methods=["POST"])
def select_model():
    """
    For simplicity, let's just store the selected algorithm in a session variable or memory.
    We'll assume the user chooses 'random-search' after training, or in general.
    """
    data = request.json
    algorithm = data.get("algorithm", "")
    if not algorithm:
        return jsonify({"message": "No algorithm selected"}), 400

    # In a more complex scenario, we might check if the model is actually trained or not.
    return jsonify({"message": f"Model {algorithm} is selected and ready to play!"})

@app.route("/model-move", methods=["POST"])
def model_move():
    """
    Receives the current board state from the front-end,
    calls predict_move to get the chosen action, 
    returns the action (index) to the client.
    """
    data = request.json
    algorithm = data.get("algorithm", "")
    board_state = data.get("board", [])
    
    if not algorithm:
        return jsonify({"error": "No algorithm specified"}), 400

    move = predict_move(algorithm, board_state)
    if move is None:
        return jsonify({"error": "No valid move found or no model loaded."}), 400

    return jsonify({"move": move})

if __name__ == "__main__":
    os.makedirs("saved_models", exist_ok=True)
    app.run(debug=True)
