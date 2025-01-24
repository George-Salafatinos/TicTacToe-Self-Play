# training_types.py

from learning import play_against_self, play_against_random, play_against_opponent_agent

def train_against_self(agent, num_games):
    metrics = []
    training_losses = []  # Store the neural network training losses
    
    for game in range(num_games):
        exp_player, won = play_against_self(agent)
        
        # Update only the main agent's policy based on its moves
        for state, action, next_state, done in exp_player:
            reward = 1 if won else -1
            loss = agent.update_policy(state, action, reward)  # Get training loss
            training_losses.append(loss)
            
        # Log game outcome metrics
        metrics.append({
            'game': game,
            'winner': won,
            'training_loss': training_losses[-1]  # Store most recent training loss
        })
        
        if game % 100 == 0:
            print(f"Self-play game {game}/{num_games}")
            
    return metrics

def train_against_random(agent, num_games):
    metrics = []
    training_losses = []
    
    for game in range(num_games):
        [experiences], won = play_against_random(agent)
        
        # Update agent's policy
        for state, action, next_state, done in experiences:
            reward = 1 if won else -1
            loss = agent.update_policy(state, action, reward)  # Get training loss
            training_losses.append(loss)
            
        # Log metrics
        metrics.append({
            'game': game,
            'won': won,
            'training_loss': training_losses[-1]
        })
        
        if game % 100 == 0:
            print(f"Random-play game {game}/{num_games}")
            
    return metrics

def train_against_opponent(agent, opponent_agent, num_games):
    metrics = []
    training_losses = []
    
    for game in range(num_games):
        [exp_agent1, exp_agent2], won = play_against_opponent_agent(agent, opponent_agent)
        
        # Update main agent's policy
        for state, action, next_state, done in exp_agent1:
            reward = 1 if won else -1
            loss = agent.update_policy(state, action, reward)  # Get training loss
            training_losses.append(loss)
            
        # Update opponent's policy
        for state, action, next_state, done in exp_agent2:
            reward = -1 if won else 1  # Opposite reward
            opponent_agent.update_policy(state, action, reward)
            
        # Log metrics
        metrics.append({
            'game': game,
            'main_agent_won': won,
            'training_loss': training_losses[-1]
        })
        
        if game % 100 == 0:
            print(f"Opponent-play game {game}/{num_games}")
            
    return metrics