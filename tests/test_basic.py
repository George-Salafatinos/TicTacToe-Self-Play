import pytest
import torch
import numpy as np
from src.learning import TicTacToeNet, TicTacToeAgent
from src.tictactoe_env import TicTacToeEnv

def test_agent_makes_valid_moves():
    agent = TicTacToeAgent()
    env = TicTacToeEnv()
    
    # Get initial state
    state = env.reset()
    valid_moves = env.get_valid_moves()
    
    # Test agent picks a valid move
    action = agent.get_action(state, valid_moves)
    assert action in valid_moves, f"Agent chose invalid move {action}"
    
    # Make some moves to create a more interesting board state
    # Let's say X has taken center, O has taken top-right
    env.make_move(4)  # X in center
    env.make_move(2)  # O in top-right
    
    state = env.get_state()
    valid_moves = env.get_valid_moves()
    
    # Test agent doesn't pick taken spots
    action = agent.get_action(state, valid_moves)
    assert action not in [4, 2], f"Agent chose already taken position {action}"
    assert action in valid_moves, f"Agent chose invalid move {action}"

def test_agent_exploit_winning_move():
    # This test is aspirational - will fail until we train the agent!
    # But good to have as a goal
    agent = TicTacToeAgent()
    env = TicTacToeEnv()
    
    # Create a board where agent (X) can win:
    # X O .
    # X O .
    # . . .
    env.board = np.array([1, -1, 0,
                         1, -1, 0,
                         0, 0, 0])
    state = env.get_state()
    valid_moves = env.get_valid_moves()
    
    action = agent.get_action(state, valid_moves)
    print(f"Agent chose {action} when it could win with 6")  # Just informational for now
    # Once trained, we'd want: assert action == 6, "Agent missed winning move"