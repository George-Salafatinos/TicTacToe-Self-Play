# train.py
from training_types import train_against_self, train_against_random, train_against_opponent
from learning import TicTacToeAgent
import matplotlib.pyplot as plt
import numpy as np

def plot_combined_metrics(metrics_dict, window=100):
    """Plot combined win rates and training losses for all training types"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    colors = {'self': 'blue', 'random': 'green', 'opponent': 'red'}
    
    # Plot win rates
    for name, metrics in metrics_dict.items():
        games = [m['game'] for m in metrics]
        
        # Get wins based on the metric name used in each training type
        if name == 'self':
            wins = [1 if m['winner'] else 0 for m in metrics]
        elif name == 'random':
            wins = [1 if m['won'] else 0 for m in metrics]
        else:  # opponent
            wins = [1 if m['main_agent_won'] else 0 for m in metrics]
            
        # Calculate moving average of win rates
        wins_avg = np.convolve(wins, np.ones(window)/window, mode='valid')
        games_avg = games[window-1:]
        
        ax1.plot(games_avg, wins_avg, color=colors[name], label=f'{name} training')
        
    ax1.set_title('Game Outcomes (Win Rates)')
    ax1.set_xlabel('Game')
    ax1.set_ylabel('Win Rate (Moving Average)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot neural network training losses
    for name, metrics in metrics_dict.items():
        games = [m['game'] for m in metrics]
        losses = [m['training_loss'] for m in metrics]
        
        # Calculate moving average of training losses
        losses_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        games_avg = games[window-1:]
        
        ax2.plot(games_avg, losses_avg, color=colors[name], label=f'{name} training')
        
    ax2.set_title('Neural Network Training Loss')
    ax2.set_xlabel('Game')
    ax2.set_ylabel('Training Loss (Moving Average)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Initialize agents
    num_games = 1400
    metrics_dict = {}

    # 1. Training against self
    print("Training against self...")
    agent = TicTacToeAgent()
    metrics_self = train_against_self(agent, num_games)
    metrics_dict['self'] = metrics_self
    agent.save_weights("model_self.pth")

    # 2. Training against random
    print("\nTraining against random...")
    agent = TicTacToeAgent()  # Fresh agent
    metrics_random = train_against_random(agent, num_games)
    metrics_dict['random'] = metrics_random
    agent.save_weights("model_random.pth")

    # 3. Training against opponent
    print("\nTraining against opponent...")
    agent = TicTacToeAgent()  # Fresh agent
    opponent = TicTacToeAgent()  # Separate opponent agent
    metrics_opponent = train_against_opponent(agent, opponent, num_games)
    metrics_dict['opponent'] = metrics_opponent
    agent.save_weights("model_opponent.pth")

    # Create combined plots
    plot_combined_metrics(metrics_dict)

if __name__ == "__main__":
    main()