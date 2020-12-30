import argparse
from GameWrapper import GameWrapper
import os, sys
import utils
import players.CompetePlayer
import itertools
import csv

ref_params = {"maxVision": 7,
                            "ccWeight": 20,
                            "possessionWeight": 1,
                            "potentialScoreWeight": 0.5,}

maxVision_param = [6, 7, 8]
ccWeight_param = [3, 20, 50]
possesionWeightParam = [0.1, 1, 10]
potentialScoreWeight_param = [0.1, 0.5, 2]

param_grid = list(itertools.product(maxVision_param, ccWeight_param, possesionWeightParam, potentialScoreWeight_param))


for params in param_grid:
    h_params = {"maxVision": params[0],
                            "ccWeight": params[1],
                            "possessionWeight": params[2],
                            "potentialScoreWeight": params[3],}
                            
    win_count = 0

    board = utils.get_board_from_csv(args.board)
    player_1 = sys.modules[players.CompetePlayer].Player(game_time, penalty_score, )
    player_2 = sys.modules[players.CompetePlayer].Player(game_time, penalty_score, )

    ...

    f = open("param_grid_res.csv", "a")
    f.write(str(params) + "," + str(win_count) + "\n")
    f.close()



## add extra args to player constructor (make them default in player file)
# continue editing from here : 


# print game info to terminal
print('Starting Game!')
print(args.player1, 'VS', args.player2)
print('Board', args.board)
print('Players have', args.move_time, 'seconds to make a signle move.')
print('Each player has', game_time, 'seconds to play in a game (global game time, sum of all moves).')

# create game with the given args
game = GameWrapper(board[0], board[1], board[2], player_1=player_1, player_2=player_2,
                terminal_viz=args.terminal_viz, 
                print_game_in_terminal=not args.dont_print_game,
                time_to_make_a_move=args.move_time, 
                game_time=game_time, 
                penalty_score = args.penalty_score,
                max_fruit_score = args.max_fruit_score,
                max_fruit_time = args.max_fruit_time)

# start playing!
game.start_game()

