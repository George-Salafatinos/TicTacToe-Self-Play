from random_search_learning import RandomSearchAgent
from tictactoe_env import TicTacToeEnv
import numpy as np

def train_rs_against_self(agent, num_games):
    metrics = []
    
    for game in range(num_games):
        env = TicTacToeEnv()
        state = env.reset()
        done = False
        
        while not done:
            valid_moves = env.get_valid_moves()
            action = agent.get_action(state, valid_moves)
            state, _, done = env.make_move(action)
        
        # Update based on game result (was agent the winner?)
        won = (env.winner == env.current_player)
        agent.update_from_game(won)
        
        # Log metrics - note we don't have training loss like original
        metrics.append({
            'game': game,
            'winner': won,
            'training_loss': 0  # Placeholder for compatibility with plotting
        })
        
        if game % 100 == 0:
            print(f"Self-play game {game}/{num_games}")
            
    return metrics

def train_rs_against_random(agent, num_games):
    metrics = []
    
    for game in range(num_games):
        env = TicTacToeEnv()
        state = env.reset()
        agent_player = np.random.choice([-1,1])  # Randomly decide if agent is X or O
        done = False
        
        while not done:
            valid_moves = env.get_valid_moves()
            
            if env.current_player == agent_player:
                # Agent's turn
                action = agent.get_action(state, valid_moves)
            else:
                # Random opponent's turn
                action = np.random.choice(valid_moves)
                
            state, _, done = env.make_move(action)
        
        # Update based on whether agent won
        won = (env.winner == agent_player)
        agent.update_from_game(won)
        
        metrics.append({
            'game': game,
            'won': won,
            'training_loss': 0
        })
        
        if game % 100 == 0:
            print(f"Random-play game {game}/{num_games}")
            
    return metrics

def train_rs_against_opponent(agent, opponent_agent, num_games):
    metrics = []
    
    for game in range(num_games):
        env = TicTacToeEnv()
        state = env.reset()
        agent_player = np.random.choice([-1,1])
        done = False
        
        while not done:
            valid_moves = env.get_valid_moves()
            
            if env.current_player == agent_player:
                action = agent.get_action(state, valid_moves)
            else:
                action = opponent_agent.get_action(state, valid_moves)
                
            state, _, done = env.make_move(action)
        
        # Update both agents
        won = (env.winner == agent_player)
        agent.update_from_game(won)
        opponent_agent.update_from_game(not won)
        
        metrics.append({
            'game': game,
            'main_agent_won': won,
            'training_loss': 0
        })
        
        if game % 100 == 0:
            print(f"Opponent-play game {game}/{num_games}")
            
    return metrics