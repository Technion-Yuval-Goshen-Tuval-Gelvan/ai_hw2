"""Search Algos: MiniMax, AlphaBeta
"""
import time

import utils
from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
import numpy as np


INTERRUPT_TIME = 0.01

class State:
    def __init__(self, board, turn, fruit_remaining_turns, player_1_pos,
                 player_2_pos, player_1_score, player_2_score, direction_from_parent):
        """
        turn is 1 or 2
        """
        self.turn = turn
        self.board = board
        self.fruit_remaining_turns = fruit_remaining_turns
        self.player_1_pos = player_1_pos
        self.player_2_pos = player_2_pos
        self.player_1_score = player_1_score
        self.player_2_score = player_2_score
        self.direction_from_parent = direction_from_parent

    def is_goal(self):
        """returns true if  move can't"""
        directions = utils.get_directions()
        if self.turn == 1:
            for d in directions:
                i = self.player_1_pos[0] + d[0]
                j = self.player_1_pos[1] + d[1]

                if 0 <= i < len(self.board) and 0 <= j < len(self.board[0]) and (
                        self.board[i][j] not in [-1, 1, 2]):  # then move is legal
                    return False
        elif self.turn == 2:
            for d in directions:
                i = self.player_2_pos[0] + d[0]
                j = self.player_2_pos[1] + d[1]

                if 0 <= i < len(self.board) and 0 <= j < len(self.board[0]) and (
                        self.board[i][j] not in [-1, 1, 2]):  # then move is legal
                    return False
        # all moves for player were iligeal :
        return True


def successor_states(cur_state):
    """
        yields successor states
    """
    directions = utils.get_directions()
    if cur_state.turn == 1:
        for d in directions:
            i = cur_state.player_1_pos[0] + d[0]
            j = cur_state.player_1_pos[1] + d[1]

            if 0 <= i < len(cur_state.board) and 0 <= j < len(cur_state.board[0]) and (
                    cur_state.board[i][j] not in [-1, 1, 2]):  # then move is legal
                additional_score = cur_state.board[i, j]  # zero if there is no fruit
                new_board = np.copy(cur_state.board)
                new_board[cur_state.player_1_pos] = -1
                new_board[i, j] = 1

                if cur_state.fruit_remaining_turns == 1:
                    new_board[new_board > 2] = 0

                yield State(new_board, 2, (cur_state.fruit_remaining_turns-1), (i,j), cur_state.player_2_pos,
                            (cur_state.player_1_score + additional_score), cur_state.player_2_score, d)
    elif cur_state.turn == 2:
        for d in directions:
            i = cur_state.player_2_pos[0] + d[0]
            j = cur_state.player_2_pos[1] + d[1]

            if 0 <= i < len(cur_state.board) and 0 <= j < len(cur_state.board[0]) and (
                    cur_state.board[i][j] not in [-1, 1, 2]):  # then move is legal
                additional_score = cur_state.board[i, j]  # zero if there is no fruit
                new_board = np.copy(cur_state.board)
                new_board[cur_state.player_2_pos] = -1
                new_board[i, j] = 2

                if cur_state.fruit_remaining_turns == 1:
                    new_board[new_board > 2] = 0

                yield State(new_board, 1, (cur_state.fruit_remaining_turns - 1), cur_state.player_1_pos, (i, j),
                            cur_state.player_1_score, (cur_state.player_2_score + additional_score), d)


def dummy_utility(state, deciding_agent):
    if deciding_agent == 1:
        return state.player_1_score
    else:
        return state.player_2_score

class SearchAlgos:
    def __init__(self, utility, succ, perform_move, goal=None):
        """The constructor for all the search algos.
        You can code these functions as you like to, 
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The succesor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = utility
        self.succ = succ
        self.perform_move = perform_move

    def search(self, state, depth, maximizing_player):
        pass


class MiniMax(SearchAlgos):

    def search(self, state, depth, maximizing_player, remaining_time):
        """Start the MiniMax algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: 1 or 2
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode, isInterrupted)
        """
        if remaining_time < INTERRUPT_TIME:
            return None, None, True

        time_this_search_start = time.time()

        if state.is_goal():
            if maximizing_player == 1:
                return state.player_1_score, None, False
            else:
                return state.player_2_score, None, False

        if depth == 0:
            return self.utility(state, maximizing_player), None, False

        agent_to_move = state.turn

        if agent_to_move == maximizing_player:
            cur_max = -np.math.inf
            level_max_direction = None
            for child in successor_states(state):

                v_cost, _, is_interrupted = self.search(child, depth-1, maximizing_player,
                                        remaining_time - (time.time() - time_this_search_start))
                if is_interrupted:
                    return None, None, True
                if v_cost > cur_max:
                    cur_max = v_cost
                    level_max_direction = child.direction_from_parent
            return cur_max, level_max_direction, False
        else:
            cur_min = np.math.inf
            for child in successor_states(state):
                v_cost, _, is_interrupted = self.search(child, depth-1, maximizing_player,
                                        remaining_time - (time.time() - time_this_search_start))
                if is_interrupted:
                    return None, None, True
                if v_cost < cur_min:
                    cur_min = v_cost
            return cur_min, None, False


class AlphaBeta(SearchAlgos):

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        #TODO: erase the following line and implement this function.
        raise NotImplementedError
