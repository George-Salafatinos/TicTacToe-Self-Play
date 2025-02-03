console.log("tictactoe.js loaded");

// Training function triggered by "Train" button
async function trainModel() {
  const algorithmSelect = document.getElementById("algorithm-select");
  const stepsInput = document.getElementById("steps-input");
  const trainStatus = document.getElementById("train-status");

  // Get user selections
  const selectedAlgorithm = algorithmSelect.value;
  const steps = parseInt(stepsInput.value) || 10; // default to 10 if empty

  if (!selectedAlgorithm) {
    trainStatus.textContent = "Please select an algorithm before training.";
    return;
  }

  trainStatus.textContent = "Training in progress...";

  try {
    // Send POST request to Flask
    const response = await fetch("/train-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        algorithm: selectedAlgorithm,
        hyperparams: {
          steps: steps
        }
      })
    });

    const data = await response.json();
    trainStatus.textContent = data.message || "Training complete (no details).";
    console.log("Training details:", data.details);
  } catch (error) {
    console.error("Error during training:", error);
    trainStatus.textContent = "Error during training. See console for details.";
  }
}

// Attach event listener to the "Train" button once the page loads
window.addEventListener("DOMContentLoaded", () => {
  const trainButton = document.getElementById("train-button");
  trainButton.addEventListener("click", trainModel);
});
