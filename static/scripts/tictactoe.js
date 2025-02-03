console.log("tictactoe.js loaded");

// Training function triggered by "Train" button
async function trainModel() {
  const algorithmSelect = document.getElementById("algorithm-select");
  const stepsInput = document.getElementById("steps-input");
  const trainStatus = document.getElementById("train-status");
  const trainingChart = document.getElementById("training-chart");

  const selectedAlgorithm = algorithmSelect.value;
  const steps = parseInt(stepsInput.value) || 10;

  if (!selectedAlgorithm) {
    trainStatus.textContent = "Please select an algorithm before training.";
    return;
  }

  trainStatus.textContent = "Training in progress...";
  trainingChart.style.display = "none"; // hide the chart if it was visible before

  try {
    const response = await fetch("/train-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        algorithm: selectedAlgorithm,
        hyperparams: { steps: steps }
      })
    });

    const data = await response.json();
    trainStatus.textContent = data.message || "Training complete (no details).";

    // If we got a base64 chart, display it
    if (data.chart_b64) {
      trainingChart.src = "data:image/png;base64," + data.chart_b64;
      trainingChart.style.display = "block";
    }

    console.log("Training details:", data.details);
  } catch (error) {
    console.error("Error during training:", error);
    trainStatus.textContent = "Error during training. See console for details.";
  }
}

// Attach event listener to the "Train" button
window.addEventListener("DOMContentLoaded", () => {
  const trainButton = document.getElementById("train-button");
  trainButton.addEventListener("click", trainModel);
});
