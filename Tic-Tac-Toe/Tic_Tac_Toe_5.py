"""
README:
========

**Game Description:**
This program implements a 5x5 Tic-Tac-Toe game where you can play against an AI, watch AI vs. AI games, or choose to let the AI play as both players. The AI uses Q-learning, a reinforcement learning technique, to learn and improve its strategies over time.

**How to Play:**
1. Run the program using the command: `python tic_tac_toe_rl.py`.
2. You will be prompted to choose between the following options:
   - 1: Human vs. AI
   - 2: AI vs. AI (watch the AI play against itself)
3. If you choose Human vs. AI, you will then be prompted to:
   - Select who plays first: 1 for Human, 2 for AI.
   - Choose your symbol: 'X' or 'O'.
4. The game will start in a window where you can make your moves by clicking on the grid.

**Additional Information:**
- The AI starts with random moves and learns over multiple games.
- The game is won by aligning 5 of your symbols in a row, column, or diagonal.

**Requirements:**
- Python 3.6 or higher
- Tkinter library (usually included with standard Python installations)
- Numpy library (install using `pip install numpy`)

**Developed by:**
Art Casasa 
"""

import numpy as np
import random
import tkinter as tk
from tkinter import messagebox

# Constants for the game
BOARD_SIZE = 5  # The game board is 5x5, as specified in the term project
EXPLORATION_RATE = 1.0  # Initial exploration rate for the AI's Q-learning
EXPLORATION_DECAY = 0.995  # Decay rate for exploration to gradually favor exploitation
LEARNING_RATE = 0.1  # Learning rate for updating Q-values
DISCOUNT_FACTOR = 0.9  # Discount factor for future rewards
EPISODES = 1000  # Reduced number of training episodes for quicker training

# Q-Table for storing state-action values
Q = {}

# Initialize the Q-table with all possible states
def initialize_q_table(board):
    """
    Initializes the Q-table with a given board state if not already present.
    Each state-action pair is represented by a unique board configuration (state)
    and the potential actions (moves) from that state.
    """
    state = board_to_tuple(board)
    if state not in Q:
        Q[state] = np.zeros(BOARD_SIZE * BOARD_SIZE)  # 25 possible actions on a 5x5 board

# Convert board to a hashable tuple for the Q-table
def board_to_tuple(board):
    """
    Converts the 2D list representing the board into a tuple of tuples.
    This transformation is necessary to use the board as a key in the Q-table dictionary.
    """
    return tuple(tuple(row) for row in board)

# Check if a player has won the game
def is_winner(board, player):
    """
    Checks if the given player has won the game by getting 5 of their symbols in a row,
    column, or diagonal.
    """
    # Check rows and columns
    for i in range(BOARD_SIZE):
        if all(board[i][j] == player for j in range(BOARD_SIZE)) or all(board[j][i] == player for j in range(BOARD_SIZE)):
            return True
    # Check diagonals
    if all(board[i][i] == player for i in range(BOARD_SIZE)) or all(board[i][BOARD_SIZE-i-1] == player for i in range(BOARD_SIZE)):
        return True
    return False

# Check if the game has ended in a draw
def is_draw(board):
    """
    Returns True if the board is full and no player has won, indicating a draw.
    """
    return all(all(cell != ' ' for cell in row) for row in board)

# Choose an action using the epsilon-greedy strategy
def choose_action(state, exploration_rate):
    """
    Selects an action using an epsilon-greedy strategy, balancing exploration and exploitation.
    If a randomly generated number is less than the exploration rate, a random move is chosen
    (exploration); otherwise, the best known move from the Q-table is selected (exploitation).
    """
    if random.uniform(0, 1) < exploration_rate:
        return random.randint(0, BOARD_SIZE * BOARD_SIZE - 1)  # Explore: random action
    else:
        return np.argmax(Q[state])  # Exploit: best action based on Q-values

# Update the Q-values based on the reward received
def update_q_table(state, action, reward, next_state):
    """
    Updates the Q-table using the Q-learning formula. The Q-value for the state-action pair is updated
    based on the observed reward and the maximum future reward possible from the next state.
    """
    best_next_action = np.argmax(Q[next_state])
    td_target = reward + DISCOUNT_FACTOR * Q[next_state][best_next_action]
    td_error = td_target - Q[state][action]
    Q[state][action] += LEARNING_RATE * td_error

# Train the AI using Q-learning over multiple episodes
def train_ai(episodes):
    """
    Trains the AI over a specified number of episodes, allowing it to explore and learn the game.
    During training, the AI explores various strategies by playing games against itself,
    updating the Q-values in the Q-table to reflect better actions.
    """
    global EXPLORATION_RATE
    for episode in range(episodes):
        # Reset the game board
        board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        initialize_q_table(board)
        state = board_to_tuple(board)
        done = False
        current_player = 'O'  # AI starts first in the training

        while not done:
            action = choose_action(state, EXPLORATION_RATE)
            row, col = divmod(action, BOARD_SIZE)

            if board[row][col] == ' ':
                board[row][col] = current_player
                reward = 0

                # Check for win/loss/draw conditions
                if is_winner(board, current_player):
                    reward = 1 if current_player == 'O' else -1
                    done = True
                elif is_draw(board):
                    reward = 0
                    done = True

                # Update Q-table
                next_state = board_to_tuple(board)
                initialize_q_table(board)
                update_q_table(state, action, reward, next_state)

                state = next_state
                current_player = 'X' if current_player == 'O' else 'O'

        # Decay the exploration rate to gradually shift from exploration to exploitation
        EXPLORATION_RATE *= EXPLORATION_DECAY

# Initialize the GUI and game logic
class TicTacToeGame:
    def __init__(self, root, player_symbol, ai_symbol, human_starts):
        """
        Initializes the Tic-Tac-Toe game GUI. Sets up the board, symbols for the player and AI,
        and determines who starts first.
        """
        self.root = root
        self.root.title("5x5 Tic-Tac-Toe with Q-learning AI")
        self.board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.human_symbol = player_symbol
        self.ai_symbol = ai_symbol
        self.current_player = player_symbol if human_starts else ai_symbol
        self.buttons = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.create_buttons()
        if not human_starts:
            self.ai_move()

    # Create buttons for each cell in the board
    def create_buttons(self):
        """
        Creates a button for each cell in the 5x5 Tic-Tac-Toe board. Each button allows the
        player to make a move by clicking on it. The buttons are arranged in a grid layout.
        """
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                self.buttons[i][j] = tk.Button(self.root, text=' ', font='Arial 20 bold', width=5, height=2,
                                               command=lambda row=i, col=j: self.make_move(row, col))
                self.buttons[i][j].grid(row=i, column=j)

    # Make a move based on the player's choice
    def make_move(self, row, col):
        """
        Handles the logic for a player's move. Updates the board and the corresponding button
        if the move is valid. Checks for a win or draw condition after the move.
        """
        if self.board[row][col] == ' ' and self.current_player == self.human_symbol:
            self.board[row][col] = self.human_symbol
            self.buttons[row][col].config(text=self.human_symbol)
            if is_winner(self.board, self.human_symbol):
                self.end_game(f"Player {self.human_symbol} wins!")
            elif is_draw(self.board):
                self.end_game("It's a draw!")
            else:
                self.current_player = self.ai_symbol
                self.ai_move()

    # The AI makes its move based on the Q-learning strategy
    def ai_move(self):
        """
        Executes the AI's move using the best action determined by the Q-table. Updates the
        board and the corresponding button. Checks for a win or draw condition after the move.
        """
        state = board_to_tuple(self.board)
        action = np.argmax(Q[state])
        row, col = divmod(action, BOARD_SIZE)

        if self.board[row][col] == ' ':
            self.board[row][col] = self.ai_symbol
            self.buttons[row][col].config(text=self.ai_symbol)
            if is_winner(self.board, self.ai_symbol):
                self.end_game(f"AI ({self.ai_symbol}) wins!")
            elif is_draw(self.board):
                self.end_game("It's a draw!")
            else:
                self.current_player = self.human_symbol

    # End the game and show a message
    def end_game(self, message):
        """
        Displays the game result in a message box and resets the game for a new round.
        This function is called when the game ends in a win or a draw.
        """
        messagebox.showinfo("Game Over", message)
        self.reset_game()

    # Reset the game to the initial state
    def reset_game(self):
        """
        Resets the game board and GUI elements to their initial state, allowing for a new game
        to begin. The current player is set based on the previous game's outcome.
        """
        self.board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                self.buttons[i][j].config(text=' ')
        self.current_player = self.human_symbol
        if self.current_player == self.ai_symbol:
            self.ai_move()

# Main function to start the game
if __name__ == "__main__":
    # Train the AI with Q-learning
    train_ai(EPISODES)

    # Game Mode Selection
    print("Select Game Mode:")
    print("1: Human vs AI")
    print("2: AI vs AI")
    game_mode = int(input("Enter choice (1 or 2): "))

    if game_mode == 1:
        # Human vs AI setup
        human_symbol = input("Choose your symbol (X or O): ").upper()
        ai_symbol = 'O' if human_symbol == 'X' else 'X'
        human_starts = int(input("Who starts first? (1: Human, 2: AI): ")) == 1
        root = tk.Tk()
        game = TicTacToeGame(root, human_symbol, ai_symbol, human_starts)
        root.mainloop()
    elif game_mode == 2:
        # AI vs AI setup
        root = tk.Tk()
        human_symbol = 'X'
        ai_symbol = 'O'
        game = TicTacToeGame(root, human_symbol, ai_symbol, False)  # Start with AI
        root.mainloop()
    else:
        print("Invalid choice. Please restart the program and choose 1 or 2.")
