import numpy as np

class Board:
    def __init__(self):
        self.state = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1  # 1 for X, -1 for O
        
    def get_state(self):
        """Returns the board state as a flat array for ML input"""
        return self.state.flatten()
    
    def get_valid_moves(self):
        """Returns list of valid move indices"""
        return np.where(self.state.flatten() == 0)[0]
    
    def make_move(self, position):
        """Makes a move if valid, returns success boolean"""
        row, col = position // 3, position % 3
        
        if self.state[row, col] != 0:
            return False
            
        self.state[row, col] = self.current_player
        self.current_player *= -1
        return True
    
    def check_winner(self):
        """Returns 1 for X win, -1 for O win, 0 for no winner yet, None for draw"""
        # Check rows and columns
        for i in range(3):
            if abs(sum(self.state[i, :])) == 3:
                return np.sign(sum(self.state[i, :]))
            if abs(sum(self.state[:, i])) == 3:
                return np.sign(sum(self.state[:, i]))
        
        # Check diagonals
        diag_sum = np.trace(self.state)
        anti_diag_sum = np.trace(np.fliplr(self.state))
        if abs(diag_sum) == 3:
            return np.sign(diag_sum)
        if abs(anti_diag_sum) == 3:
            return np.sign(anti_diag_sum)
        
        # Check for draw
        if len(self.get_valid_moves()) == 0:
            return None
            
        return 0  # Game ongoing
    
    def is_terminal(self):
        """Returns True if game is over"""
        return self.check_winner() is not None or len(self.get_valid_moves()) == 0
    
    def reset(self):
        """Resets the board to initial state"""
        self.state = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1