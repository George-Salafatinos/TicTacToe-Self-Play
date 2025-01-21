# tictactoe_env.py

import numpy as np

class TicTacToeEnv:
    def __init__(self):
        # Board state: 0 = empty, 1 = X, -1 = O
        self.board = np.zeros(9)
        # Current player: 1 = X, -1 = O
        self.current_player = 1
        self.done = False
        self.winner = None
    
    def reset(self):
        """Reset the game to initial state"""
        self.board = np.zeros(9)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()
    
    def get_state(self):
        """Return current state (board + current player)"""
        return np.append(self.board, self.current_player)
    
    def get_valid_moves(self):
        """Return list of valid move indices"""
        return np.where(self.board == 0)[0]
    
    def make_move(self, position):
        """Make a move on the board. Returns (new_state, reward, done)"""
        if self.done:
            return self.get_state(), 0, True
            
        if position not in self.get_valid_moves():
            return self.get_state(), -10, True  # Invalid move penalty
            
        # Make the move
        self.board[position] = self.current_player
        
        # Check for win/draw
        if self._check_win():
            self.done = True
            self.winner = self.current_player
            return self.get_state(), 1, True  # Win
            
        if len(self.get_valid_moves()) == 0:
            self.done = True
            return self.get_state(), 0, True  # Draw
            
        # Switch players
        self.current_player *= -1
        return self.get_state(), 0, False
    
    def _check_win(self):
        """Check if current player has won"""
        # Define winning combinations
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        
        player = self.current_player
        board = self.board.reshape(3, 3)
        
        for combo in win_combinations:
            if all(self.board[i] == player for i in combo):
                return True
        return False
    
    def render(self):
        """Print current board state"""
        symbols = {0: ".", 1: "X", -1: "O"}
        board = self.board.reshape(3, 3)
        for row in board:
            print(" ".join(symbols[x] for x in row))
        print()