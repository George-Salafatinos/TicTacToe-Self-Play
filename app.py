from flask import Flask, render_template, request, jsonify
import os

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

    result = train_selected_model(algorithm, hyperparams)

    # If there's a chart in base64, add it to the JSON response
    chart_b64 = result.get("chart_b64", None)

    return jsonify({
        "message": "Training complete!",
        "details": result,
        "chart_b64": chart_b64
    })

if __name__ == "__main__":
    os.makedirs("saved_models", exist_ok=True)
    app.run(debug=True)
