document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const trainButton = document.getElementById('train-button');
    const modelSelect = document.getElementById('model-select');
    const cells = document.querySelectorAll('.cell');

    // Training button click handler (to be implemented)
    trainButton.addEventListener('click', () => {
        const algorithm = document.getElementById('algorithm').value;
        const opponent = document.getElementById('opponent').value;
        const episodes = document.getElementById('episodes').value;
        
        console.log('Training settings:', { algorithm, opponent, episodes });
        // Training endpoint will be implemented in next sprint
    });

    // Disable board interaction for now
    cells.forEach(cell => {
        cell.style.cursor = 'not-allowed';
    });
});