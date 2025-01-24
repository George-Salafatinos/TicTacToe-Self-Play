from learning import TicTacToeAgent
from random_search_learning import RandomSearchAgent
from training_types import train_against_self, train_against_random, train_against_opponent
from random_search_training import train_rs_against_self, train_rs_against_random, train_rs_against_opponent
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(reinforce_metrics, rs_metrics, title, window=100):
    plt.figure(figsize=(10, 6))
    
    # Get games and wins for both methods
    games_reinforce = [m['game'] for m in reinforce_metrics]
    if 'winner' in reinforce_metrics[0]:
        wins_reinforce = [1 if m['winner'] else 0 for m in reinforce_metrics]
    elif 'won' in reinforce_metrics[0]:
        wins_reinforce = [1 if m['won'] else 0 for m in reinforce_metrics]
    else:
        wins_reinforce = [1 if m['main_agent_won'] else 0 for m in reinforce_metrics]
        
    games_rs = [m['game'] for m in rs_metrics]
    if 'winner' in rs_metrics[0]:
        wins_rs = [1 if m['winner'] else 0 for m in rs_metrics]
    elif 'won' in rs_metrics[0]:
        wins_rs = [1 if m['won'] else 0 for m in rs_metrics]
    else:
        wins_rs = [1 if m['main_agent_won'] else 0 for m in rs_metrics]
    
    # Calculate moving averages
    wins_avg_reinforce = np.convolve(wins_reinforce, np.ones(window)/window, mode='valid')
    wins_avg_rs = np.convolve(wins_rs, np.ones(window)/window, mode='valid')
    
    games_avg_reinforce = games_reinforce[window-1:]
    games_avg_rs = games_rs[window-1:]
    
    plt.plot(games_avg_reinforce, wins_avg_reinforce, label='REINFORCE', color='blue')
    plt.plot(games_avg_rs, wins_avg_rs, label='Random Search', color='red')
    
    plt.title(f'{title} - Win Rate Comparison')
    plt.xlabel('Game')
    plt.ylabel('Win Rate (Moving Average)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'comparison_{title.lower().replace(" ", "_")}.png')
    plt.close()

def main():
    num_games = 3000
    
    # Test against self
    print("Training both methods against self...")
    reinforce_agent = TicTacToeAgent()
    rs_agent = RandomSearchAgent()
    
    metrics_reinforce_self = train_against_self(reinforce_agent, num_games)
    metrics_rs_self = train_rs_against_self(rs_agent, num_games)
    plot_comparison(metrics_reinforce_self, metrics_rs_self, "Self Play")
    
    # Test against random
    print("\nTraining both methods against random...")
    reinforce_agent = TicTacToeAgent()
    rs_agent = RandomSearchAgent()
    
    metrics_reinforce_random = train_against_random(reinforce_agent, num_games)
    metrics_rs_random = train_rs_against_random(rs_agent, num_games)
    plot_comparison(metrics_reinforce_random, metrics_rs_random, "Random Opponent")
    
    # Test against opponent
    print("\nTraining both methods against opponent...")
    reinforce_agent = TicTacToeAgent()
    reinforce_opponent = TicTacToeAgent()
    rs_agent = RandomSearchAgent()
    rs_opponent = RandomSearchAgent()
    
    metrics_reinforce_opp = train_against_opponent(reinforce_agent, reinforce_opponent, num_games)
    metrics_rs_opp = train_rs_against_opponent(rs_agent, rs_opponent, num_games)
    plot_comparison(metrics_reinforce_opp, metrics_rs_opp, "Training Opponent")

if __name__ == "__main__":
    main()