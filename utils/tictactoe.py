def check_winner(board):
    """
    board: list of 9 elements, each 'X', 'O', or '' for empty
    Return 'X' or 'O' if there's a winner, or None if no winner yet.
    """
    win_patterns = [
        (0,1,2), (3,4,5), (6,7,8), # rows
        (0,3,6), (1,4,7), (2,5,8), # cols
        (0,4,8), (2,4,6)           # diagonals
    ]
    for a, b, c in win_patterns:
        if board[a] == board[b] == board[c] and board[a] != '':
            return board[a]
    return None

def available_moves(board):
    """
    Return a list of indices where the board is empty ('').
    """
    return [i for i, cell in enumerate(board) if cell == '']

class TicTacToeEnv:
    """
    Minimal Tic-Tac-Toe environment-like class (not fully gym-compliant).
    We'll expand or adjust as we add more logic in future sprints.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        # Represent the board as a list of 9, each either '', 'X', or 'O'
        self.board = [''] * 9
        self.current_player = 'X'
        return self.board

    def step(self, action):
        """
        action: index (0-8) where the current player places 'X' or 'O'.
        Returns: next_state, reward, done, info
        """
        if self.board[action] != '':
            # Invalid move, return penalty or handle differently
            return self.board, -1, True, {}

        # Place the current player's mark
        self.board[action] = self.current_player

        winner = check_winner(self.board)
        if winner:
            # Current player has won
            return self.board, 1, True, {"winner": winner}
        elif '' not in self.board:
            # It's a draw
            return self.board, 0, True, {"winner": None}
        else:
            # Game continues, switch player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return self.board, 0, False, {}

    def is_done(self):
        """
        Return True if the game ended (win or draw).
        """
        if check_winner(self.board) is not None:
            return True
        if '' not in self.board:
            return True
        return False
