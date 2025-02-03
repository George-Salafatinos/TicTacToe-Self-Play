console.log("tictactoe.js loaded");

let currentBoard = ["", "", "", "", "", "", "", "", ""];
let currentPlayer = "X"; // We'll assume user starts as 'X'
let selectedAlgorithm = "";

// Training function triggered by "Train" button
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

// SELECT MODEL
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

// Click handler for board cells
function onCellClick(event) {
  const cell = event.target;
  const index = parseInt(cell.dataset.index);

  // If cell already occupied or it's not X's turn, do nothing
  if (currentBoard[index] !== "" || currentPlayer !== "X") {
    return;
  }

  // Place 'X'
  currentBoard[index] = "X";
  cell.textContent = "X";

  // Check if the game is over or we need O's move
  if (!isGameOver()) {
    currentPlayer = "O";
    modelMove();
  }
}

// Ask the model to move
async function modelMove() {
  const playStatus = document.getElementById("play-status");

  // Send current board to server
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
      currentPlayer = "X"; // revert turn so user can continue
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

    // Check if game is over
    if (!isGameOver()) {
      currentPlayer = "X";
    }

  } catch (error) {
    console.error("Error during model move:", error);
    playStatus.textContent = "Error during model move. See console.";
    currentPlayer = "X"; // revert turn
  }
}

// Basic function to check if game is over
function isGameOver() {
  const playStatus = document.getElementById("play-status");
  // check winner or draw
  const winner = checkWinner(currentBoard);
  if (winner) {
    playStatus.textContent = `Game Over! Winner: ${winner}`;
    return true;
  }
  if (!currentBoard.includes("")) {
    playStatus.textContent = `Game Over! It's a draw.`;
    return true;
  }
  return false;
}

// Basic logic for checking winner on front-end
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

// Attach event listeners after DOM loads
window.addEventListener("DOMContentLoaded", () => {
  const trainButton = document.getElementById("train-button");
  trainButton.addEventListener("click", trainModel);

  const selectModelButton = document.getElementById("select-model-button");
  selectModelButton.addEventListener("click", selectModel);

  // Board clicks
  const cells = document.querySelectorAll(".cell");
  cells.forEach(cell => {
    cell.addEventListener("click", onCellClick);
  });
});
