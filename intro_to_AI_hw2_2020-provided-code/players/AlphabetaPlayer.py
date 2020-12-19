"""
MiniMax Player with AlphaBeta pruning
"""
import time

from players.AbstractPlayer import AbstractPlayer
import numpy as np
from SearchAlgos import State, AlphaBeta, sum_heuristic, successor_states


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time,
                                penalty_score)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.board = None
        self.board_min_len = -1
        self.played_turns = 0
        self.player_pos = (-1, -1)
        self.rival_pos = (-1, -1)
        self.player_score = 0
        self.rival_score = 0

        self.heuristic = sum_heuristic
        self.algorithm = AlphaBeta(self.heuristic, successor_states, None)

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        self.board = np.copy(board)
        self.board_min_len = np.min(len(board))

        tmp_player_pos = np.where(board == 1)
        self.player_pos = (tmp_player_pos[0][0], tmp_player_pos[1][0])
        tmp_rival_pos = np.where(board == 2)
        self.rival_pos = (tmp_rival_pos[0][0], tmp_rival_pos[1][0])

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        # print("turn", self.played_turns)
        # print("min len", self.board_min_len)
        # print(self.board)

        current_state = State(self.board, 1, self.board_min_len - self.played_turns, self.player_pos,
                              self.rival_pos, self.player_score, self.rival_score, None)

        depth = 1
        remaining_time = time_limit
        is_interrupted = False
        tmp_chosen_direction = None
        while not is_interrupted and depth < 50:
            t = time.time()
            h_val, tmp_chosen_direction, is_interrupted = self.algorithm.search(current_state, depth, 1,
                                                                                remaining_time, self.penalty_score)
            if not is_interrupted:
                chosen_direction = tmp_chosen_direction
            remaining_time -= (time.time() - t)
            # print("remaining time:", remaining_time)
            depth += 1

        # print("h_val" , h_val)
        print("searched depth :", depth - 1)

        new_player_pos = (self.player_pos[0] + chosen_direction[0], self.player_pos[1] + chosen_direction[1])
        self.board[self.player_pos] = -1
        self.player_score += self.board[new_player_pos]
        self.player_pos = new_player_pos
        self.board[new_player_pos] = 1

        # print("rival_score", self.rival_score)
        # print("player_score", self.player_score)
        return chosen_direction[0], chosen_direction[1]

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        rival_id = self.board[self.rival_pos]
        self.board[self.rival_pos] = -1
        self.rival_score += self.board[pos]
        self.rival_pos = pos
        self.board[pos] = rival_id

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        self.played_turns += 1
        if self.played_turns == (self.board_min_len + 1):
            # time to clear fruits
            self.board[self.board > 2] = 0

        if self.played_turns < 2:
            for key in fruits_on_board_dict.keys():
                self.board[key] = fruits_on_board_dict[key]


    ########## helper functions in class ##########
    #TODO: add here helper functions in class, if needed


    ########## helper functions for AlphaBeta algorithm ##########
    #TODO: add here the utility, succ, and perform_move functions used in AlphaBeta algorithm