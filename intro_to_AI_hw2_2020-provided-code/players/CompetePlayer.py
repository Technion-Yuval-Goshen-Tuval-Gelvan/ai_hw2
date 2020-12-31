"""
MiniMax Player with AlphaBeta pruning and global time
"""
import time

from players.AbstractPlayer import AbstractPlayer
import numpy as np
from SearchAlgos import State, CompeteAlgo, successor_states, compete_heuristic, connected_components_heuristic

default_heuristic_params = {"maxVision": 7,
                            "ccWeight": 50,
                            "possessionWeight": 1,
                            "potentialScoreWeight": 0.5,}


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score, heuristic_params=default_heuristic_params):
        AbstractPlayer.__init__(self, game_time,
                                penalty_score)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.board = None
        self.board_min_len = -1
        self.played_turns = 0
        self.player_pos = (-1, -1)
        self.rival_pos = (-1, -1)
        self.player_score = 0
        self.rival_score = 0
        self.game_remaining_time = game_time

        self.heuristic = compete_heuristic
        self.heuristic_params = heuristic_params
        self.heuristic_params["isDifferentCC"] = False
        self.algorithm = CompeteAlgo(utility=None,
                                     succ=successor_states,
                                     perform_move=None,
                                     heuristic=self.heuristic)

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
        turn_start_time = time.time()

        current_state = State(self.board, 1, self.board_min_len - self.played_turns, self.player_pos,
                              self.rival_pos, self.player_score, self.rival_score, None)

        if current_state.fruit_remaining_turns > 0:  # fruit stage
            # calculate average time for turn assuming all squares are reachable
            expected_remaining_turns = len(np.argwhere(np.logical_or(
                self.board == 0, self.board > 2))) / 2
            avg_turn_time = self.game_remaining_time / expected_remaining_turns
            turn_time_limit = avg_turn_time
            if self.played_turns < (self.board_min_len / 4):
                turn_time_limit *= 4

        else:  # no fruit stage
            # calculate average time for turn, use the following heuristic because it
            # it does exactly what we need
            player_cc, rival_cc, is_same_cc = connected_components_heuristic(current_state, 1)
            expected_remaining_turns = np.min([player_cc, rival_cc])
            avg_turn_time = avg_turn_time = self.game_remaining_time / expected_remaining_turns
            turn_time_limit = avg_turn_time

            if not is_same_cc:
                self.heuristic_params["isDifferentCC"] = True

            # give more time to do the best in the beginning of the stage if its not to dangarous:
            if current_state.fruit_remaining_turns > -2 and expected_remaining_turns >= 4:
                turn_time_limit *= 3
            
        print("avg remaining turn time", avg_turn_time, "(compete)")
        depth = 1
        remaining_turn_time = turn_time_limit
        last_iteration_time = 0
        chosen_direction = (0, 1)
        while True:
            t = time.time()
            chosen_h, chosen_direction = self.algorithm.search(state=current_state,
                                                               depth=depth,
                                                               maximizing_player=1,
                                                               penalty=self.penalty_score,
                                                               heuristic_params=self.heuristic_params)
            last_iteration_time = time.time() - t
            remaining_turn_time -= last_iteration_time
            depth += 1
            if not(remaining_turn_time > last_iteration_time * 3 and depth < 50):
                break

        print("searched depth :", depth - 1, "(compete)")
        new_player_pos = (self.player_pos[0] + chosen_direction[0], self.player_pos[1] + chosen_direction[1])
        self.board[self.player_pos] = -1
        self.player_score += self.board[new_player_pos]
        self.player_pos = new_player_pos
        self.board[new_player_pos] = 1

        self.game_remaining_time -= time.time() - turn_start_time
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
