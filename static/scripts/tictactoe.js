console.log("tictactoe.js loaded");

let currentBoard = ["", "", "", "", "", "", "", "", ""];
let currentPlayer = "X";
let selectedAlgorithm = "";

async function trainModel() {
  const algorithmSelect = document.getElementById("algorithm-select");
  const stepsInput = document.getElementById("steps-input");
  const trainStatus = document.getElementById("train-status");
  const trainingChart = document.getElementById("training-chart");

  selectedAlgorithm = algorithmSelect.value;
  const steps = parseInt(stepsInput.value) || 10;

  if (!selectedAlgorithm) {
    trainStatus.textContent = "Please select an algorithm before training.";
    return;
  }

  trainStatus.textContent = "Training in progress...";
  trainingChart.style.display = "none";

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

async function selectModel() {
  const playStatus = document.getElementById("play-status");
  if (!selectedAlgorithm) {
    playStatus.textContent = "No model trained or selected. Train first.";
    return;
  }
  try {
    const resp = await fetch("/select-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ algorithm: selectedAlgorithm })
    });
    const data = await resp.json();
    playStatus.textContent = data.message || "Model selected.";
  } catch (err) {
    console.error("Error selecting model:", err);
    playStatus.textContent = "Error selecting model. See console.";
  }
}

function onCellClick(event) {
  const cell = event.target;
  const index = parseInt(cell.dataset.index);

  if (currentBoard[index] !== "" || currentPlayer !== "X") {
    return;
  }
  currentBoard[index] = "X";
  cell.textContent = "X";

  if (!isGameOver()) {
    currentPlayer = "O";
    modelMove();
  }
}

async function modelMove() {
  const playStatus = document.getElementById("play-status");
  try {
    const response = await fetch("/model-move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        algorithm: selectedAlgorithm,
        board: currentBoard
      })
    });
    const data = await response.json();
    if (data.error) {
      playStatus.textContent = "Model move error: " + data.error;
      currentPlayer = "X";
      return;
    }
    const moveIndex = data.move;
    if (moveIndex !== null && moveIndex !== undefined) {
      currentBoard[moveIndex] = "O";
      const cell = document.getElementById(`cell-${moveIndex}`);
      if (cell) {
        cell.textContent = "O";
      }
    }
    if (!isGameOver()) {
      currentPlayer = "X";
    }
  } catch (error) {
    console.error("Error during model move:", error);
    playStatus.textContent = "Error during model move. See console.";
    currentPlayer = "X";
  }
}

function isGameOver() {
  const playStatus = document.getElementById("play-status");
  const winner = checkWinner(currentBoard);
  if (winner) {
    playStatus.textContent = "Game Over! Winner: " + winner;
    return true;
  }
  if (!currentBoard.includes("")) {
    playStatus.textContent = "Game Over! It's a draw.";
    return true;
  }
  return false;
}

function checkWinner(board) {
  const lines = [
    [0,1,2], [3,4,5], [6,7,8],
    [0,3,6], [1,4,7], [2,5,8],
    [0,4,8], [2,4,6]
  ];
  for (const [a,b,c] of lines) {
    if (board[a] && board[a] === board[b] && board[a] === board[c]) {
      return board[a];
    }
  }
  return null;
}

window.addEventListener("DOMContentLoaded", () => {
  const trainButton = document.getElementById("train-button");
  trainButton.addEventListener("click", trainModel);

  const selectModelButton = document.getElementById("select-model-button");
  selectModelButton.addEventListener("click", selectModel);

  const cells = document.querySelectorAll(".cell");
  cells.forEach(cell => {
    cell.addEventListener("click", onCellClick);
  });
});
