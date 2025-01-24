# app.py
from flask import Flask, render_template, jsonify, request
from src.tictactoe_env import TicTacToeEnv

app = Flask(__name__)

# Global game environment
game = TicTacToeEnv()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/make_move', methods=['POST'])
def make_move():
    position = int(request.json['position'])
    state, reward, done = game.make_move(position)
    return jsonify({
        'board': state[:9].tolist(),  # Convert numpy array to list
        'current_player': int(state[9]),
        'valid_moves': game.get_valid_moves().tolist(),
        'done': done,
        'reward': reward
    })

@app.route('/reset', methods=['POST'])
def reset():
    state = game.reset()
    return jsonify({
        'board': state[:9].tolist(),
        'current_player': int(state[9]),
        'valid_moves': game.get_valid_moves().tolist(),
        'done': False
    })

if __name__ == '__main__':
    app.run(debug=True)