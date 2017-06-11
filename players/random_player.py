import random
import yavalath_engine

def random_player(game_so_far):
    board = yavalath_engine.HexBoard()
    return random.choice(list(set(board.spaces) - set(game_so_far)))

def get_player_names():
    return ["random"]

def get_player(name):
    return random_player
