from flask import Flask, render_template, request, jsonify
import os

# Import training function from the new "training" folder
from training.train import train_selected_model

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train-model", methods=["POST"])
def train_model():
    """Endpoint to kick off training."""
    data = request.json
    algorithm = data.get("algorithm", "")
    hyperparams = data.get("hyperparams", {})

    if not algorithm:
        return jsonify({"message": "No algorithm specified"}), 400

    # Call training orchestrator
    result = train_selected_model(algorithm, hyperparams)

    return jsonify({
        "message": "Training complete!",
        "details": result
    })

if __name__ == "__main__":
    # Ensure training folder exists for saved models (if you save them)
    os.makedirs("saved_models", exist_ok=True)
    app.run(debug=True)
