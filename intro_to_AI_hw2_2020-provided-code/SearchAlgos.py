"""Search Algos: MiniMax, AlphaBeta
"""
import time

import utils
from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
import numpy as np
import networkx as nx


INTERRUPT_TIME = 0.25

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


def is_penalty(state):
    """
    returns 0 if there is no penalty
    1 if player 1 gets penalty
    2 if player 2 gets penalty
    """
    directions = utils.get_directions()
    can_move_1 = False
    for d in directions:
        i = state.player_1_pos[0] + d[0]
        j = state.player_1_pos[1] + d[1]

        if 0 <= i < len(state.board) and 0 <= j < len(state.board[0]) and (
                state.board[i][j] not in [-1, 1, 2]):  # then move is legal
            can_move_1 = True
            break
    can_move_2 = False
    for d in directions:
        i = state.player_2_pos[0] + d[0]
        j = state.player_2_pos[1] + d[1]

        if 0 <= i < len(state.board) and 0 <= j < len(state.board[0]) and (
                state.board[i][j] not in [-1, 1, 2]):  # then move is legal
            can_move_2 = True
            break

    if can_move_1 and not can_move_2:
        return 2
    elif not can_move_1 and  can_move_2:
        return 1
    else:
        return 0


def successor_states(cur_state, penalty):
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

                new_state = State(new_board, 2, (cur_state.fruit_remaining_turns-1), (i,j), cur_state.player_2_pos,
                            (cur_state.player_1_score + additional_score), cur_state.player_2_score, d)
                is_pen = is_penalty(new_state)
                if is_pen == 1:
                    new_state.player_1_score -= penalty
                if is_pen == 2:
                    new_state.player_2_score -= penalty
                yield new_state

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

                new_state = State(new_board, 1, (cur_state.fruit_remaining_turns - 1), cur_state.player_1_pos, (i, j),
                            cur_state.player_1_score, (cur_state.player_2_score + additional_score), d)
                is_pen = is_penalty(new_state)
                if is_pen == 1:
                    new_state.player_1_score -= penalty
                if is_pen == 2:
                    new_state.player_2_score -= penalty
                yield new_state


##################### heuristics ###################################
def man_dist(p1, p2):
    """helper func to calculate manheten distance beetween two points"""
    return np.absolute(p1[0] - p2[0]) + np.absolute(p1[1] - p2[1])


def score_heuristic(state, deciding_agent):
    """ h1 in the report"""
    if deciding_agent == 1:
        return state.player_1_score
    else:
        return state.player_2_score


def rival_score_heuristic(state, deciding_agent):
    """ h5 in the report"""
    if deciding_agent == 1:
        return -state.player_2_score
    else:
        return -state.player_1_score


def squares_in_possession_heuristic(state, deciding_agent):
    """ h3 in the report"""
    # returns list of pairs for cordinates where there are empty squares
    empty_squares_and_fruits_loc = np.argwhere(
        np.logical_or(state.board == 0, state.board > 2))

    count_closest_to_1 = sum(man_dist(sq, state.player_1_pos) < man_dist(sq, state.player_2_pos) for sq in empty_squares_and_fruits_loc)

    if deciding_agent == 1:
        return count_closest_to_1
    else:
        return len(empty_squares_and_fruits_loc) - count_closest_to_1


def potential_score_heuristic(state, deciding_agent):
    """ h3 in the report"""
    fruits_loc = np.argwhere(state.board > 2)

    if deciding_agent == 1:
        player_position = state.player_1_pos
    else:
        player_position = state.player_2_pos

    return sum(state.board[tuple(fr)]/man_dist(fr, player_position)
               for fr in fruits_loc
               if man_dist(fr, player_position) <= state.fruit_remaining_turns)


def connected_components_heuristic(state, deciding_agent):
    squares = np.argwhere(state.board != -1)

    right_edges = [(tuple(s), (s[0], s[1]+1)) for s in squares if [s[0], s[1]+1] in squares.tolist()]
    left_edges = [(tuple(s), (s[0], s[1]-1)) for s in squares if [s[0], s[1]-1] in squares.tolist()]
    up_edges = [(tuple(s), (s[0]-1, s[1])) for s in squares if [s[0]-1, s[1]] in squares.tolist()]
    down_edges = [(tuple(s), (s[0]+1, s[1])) for s in squares if [s[0]+1, s[1]] in squares.tolist()]

    G = nx.Graph(right_edges + left_edges + down_edges + up_edges)
    G.add_node(state.player_1_pos)
    G.add_node(state.player_2_pos)
    # print(state.player_1_pos, state.player_2_pos)
    # print(state.board)
    # print(right_edges + left_edges + down_edges + up_edges)

    return len(nx.node_connected_component(G, state.player_1_pos)),\
           len(nx.node_connected_component(G, state.player_2_pos))


def sum_heuristic(state, deciding_agent):
    return (score_heuristic(state, deciding_agent)
            + rival_score_heuristic(state, deciding_agent)
            + squares_in_possession_heuristic(state, deciding_agent)
            + potential_score_heuristic(state, deciding_agent))


def phases_sum_heuristic(state, deciding_agent):
    if state.fruit_remaining_turns >= 0:
        return (score_heuristic(state, deciding_agent)
            + rival_score_heuristic(state, deciding_agent)
            + potential_score_heuristic(state, deciding_agent)
            + squares_in_possession_heuristic(state, deciding_agent))
    else:
        player1_cc, player2_cc = connected_components_heuristic(state, deciding_agent)
        cc_h = player1_cc - player2_cc
        if deciding_agent == 2:
            cc_h = -cc_h
        
        return (50 * cc_h
        + squares_in_possession_heuristic(state, deciding_agent)
        + score_heuristic(state, deciding_agent)
        + rival_score_heuristic(state, deciding_agent))




def compete_heuristic(state, deciding_agent, heuristic_params):
    """
    everithing is calculated in this single function to save time
    heuristic_params is dictionary with the parameters and weights
    params:
    "isShortVersion" - calculate the short version
    "maxVision" - how far from the player to look
    """

    if deciding_agent == 1:
        player_pos = state.player_1_pos
        rival_pos = state.player_2_pos
        player_id = 1
        rival_id = 2
    else:
        player_pos = state.player_2_pos
        rival_pos = state.player_1_pos
        player_id = 2
        rival_id = 1

    # set players vision:
    # vision is set acording to the parameter but makes sure that
    # rival will be in sight and at least one block after him
    i_min = np.min([player_pos[0] - heuristic_params["maxVision"], rival_pos[0] - 1])
    i_min = np.max([0, i_min])
    i_max = np.max([player_pos[0] + heuristic_params["maxVision"], rival_pos[0] + 1])
    i_max = np.min([state.board.shape[0]-1, i_max])

    j_min = np.min([player_pos[1] - heuristic_params["maxVision"], rival_pos[1] - 1])
    j_min = np.max([0, j_min])
    j_max = np.max([player_pos[1] + heuristic_params["maxVision"], rival_pos[1] + 1])
    j_max = np.min([state.board.shape[1]-1, j_max])

    player_vision = state.board[i_min:i_max+1, j_min:j_max+1]
    player_pos_in_vision = tuple(np.argwhere(player_vision == player_id)[0])
    rival_pos_in_vision = tuple(np.argwhere(player_vision == rival_id)[0])

    # create graph:
    # TODO: only if long version

    squares = np.argwhere(player_vision != -1)

    right_edges = [(tuple(s), (s[0], s[1] + 1)) for s in squares if [s[0], s[1] + 1] in squares.tolist()]
    left_edges = [(tuple(s), (s[0], s[1] - 1)) for s in squares if [s[0], s[1] - 1] in squares.tolist()]
    up_edges = [(tuple(s), (s[0] - 1, s[1])) for s in squares if [s[0] - 1, s[1]] in squares.tolist()]
    down_edges = [(tuple(s), (s[0] + 1, s[1])) for s in squares if [s[0] + 1, s[1]] in squares.tolist()]

    G = nx.Graph(right_edges + left_edges + down_edges + up_edges)
    G.add_node(state.player_pos_in_vision)
    G.add_node(state.rival_pos_in_vision)



##################### heuristics end ################################


class SearchAlgos:
    def __init__(self, utility, succ, perform_move, heuristic=score_heuristic, goal=None):
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
        self.heuristic = heuristic

    def search(self, state, depth, maximizing_player):
        pass


class MiniMax(SearchAlgos):

    def search(self, state, depth, maximizing_player, remaining_time, penalty):
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
            return self.heuristic(state, maximizing_player), None, False

        agent_to_move = state.turn

        if agent_to_move == maximizing_player:
            cur_max = -np.math.inf
            level_max_direction = None
            for child in successor_states(state, penalty):

                v_cost, _, is_interrupted = self.search(child, depth-1, maximizing_player,
                                        remaining_time - (time.time() - time_this_search_start), penalty)
                if is_interrupted:
                    return None, None, True
                if v_cost > cur_max:
                    cur_max = v_cost
                    level_max_direction = child.direction_from_parent
            return cur_max, level_max_direction, False
        else:
            cur_min = np.math.inf
            for child in successor_states(state, penalty):
                v_cost, _, is_interrupted = self.search(child, depth-1, maximizing_player,
                                        remaining_time - (time.time() - time_this_search_start), penalty)
                if is_interrupted:
                    return None, None, True
                if v_cost < cur_min:
                    cur_min = v_cost
            return cur_min, None, False


class AlphaBeta(SearchAlgos):
    def search(self, state, depth, maximizing_player, remaining_time, penalty, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
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
            return self.heuristic(state, maximizing_player), None, False

        agent_to_move = state.turn

        if agent_to_move == maximizing_player:
            cur_max = -np.math.inf
            level_max_direction = None
            for child in successor_states(state, penalty):
                v_cost, _, is_interrupted = self.search(child, depth - 1, maximizing_player,
                                                        remaining_time - (time.time() - time_this_search_start),
                                                        penalty,
                                                        alpha,
                                                        beta)
                if is_interrupted:
                    return None, None, True
                if v_cost > cur_max:
                    cur_max = v_cost
                    level_max_direction = child.direction_from_parent
                if cur_max > alpha:
                    alpha = cur_max
                if cur_max >= beta:
                    return np.math.inf, None, False
            return cur_max, level_max_direction, False

        else:
            cur_min = np.math.inf
            for child in successor_states(state, penalty):
                v_cost, _, is_interrupted = self.search(child, depth - 1, maximizing_player,
                                                        remaining_time - (time.time() - time_this_search_start),
                                                        penalty,
                                                        alpha,
                                                        beta)
                if is_interrupted:
                    return None, None, True
                if v_cost < cur_min:
                    cur_min = v_cost
                if cur_min < beta:
                    beta = cur_min
                if cur_min <= alpha:
                    return -np.math.inf, None, False
            return cur_min, None, False


class CompeteAlgo(SearchAlgos):
    """algorithem for the Competition, mostly like AlphaBeta but uses weights"""
    def __init__(self, utility, succ, perform_move, heuristic_params, heuristic, goal=None):
        SearchAlgos.__init__(self, utility, succ, perform_move, heuristic, goal)
        self.heuristic_params = heuristic_params


    def search(self, state, depth, maximizing_player, remaining_time, penalty, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
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
            return self.heuristic(state, maximizing_player, self.heuristic_params), None, False

        agent_to_move = state.turn

        if agent_to_move == maximizing_player:
            cur_max = -np.math.inf
            level_max_direction = None
            for child in successor_states(state, penalty):
                v_cost, _, is_interrupted = self.search(child, depth - 1, maximizing_player,
                                                        remaining_time - (time.time() - time_this_search_start),
                                                        penalty,
                                                        alpha,
                                                        beta)
                if is_interrupted:
                    return None, None, True
                if v_cost > cur_max:
                    cur_max = v_cost
                    level_max_direction = child.direction_from_parent
                if cur_max > alpha:
                    alpha = cur_max
                if cur_max >= beta:
                    return np.math.inf, None, False
            return cur_max, level_max_direction, False

        else:
            cur_min = np.math.inf
            for child in successor_states(state, penalty):
                v_cost, _, is_interrupted = self.search(child, depth - 1, maximizing_player,
                                                        remaining_time - (time.time() - time_this_search_start),
                                                        penalty,
                                                        alpha,
                                                        beta)
                if is_interrupted:
                    return None, None, True
                if v_cost < cur_min:
                    cur_min = v_cost
                if cur_min < beta:
                    beta = cur_min
                if cur_min <= alpha:
                    return -np.math.inf, None, False
            return cur_min, None, False

