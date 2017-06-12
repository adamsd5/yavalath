import random
import yavalath_engine

def random_player(game_so_far):
    board = yavalath_engine.HexBoard()
    return random.choice(list(set(board.spaces) - set(game_so_far)))

def get_player_names():
    return ["random"]

def get_player(name):
    if name == "random":
        return random_player
    raise NotImplementedError("The player {} is not implemented in this module".format(name))