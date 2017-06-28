import yavalath_engine
import random
import logging
from enum import Enum
import pprint
import numpy
import itertools
import functools
import collections
import pickle
import time
import multiprocessing
import queue
import os
import pathlib
from players.blue.classifiers import SpaceProperies, NextMoveClassifier  # Needed to unpickle.
from players.blue import blue0, blue1, blue2

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

all_players = {
    "abbot": lambda moves: blue0.player(moves, depth=1),
    "costello": lambda moves: blue0.player(moves, depth=2),
    "groucho": lambda moves: blue2.player(moves, depth=1)[0],  # Returns move, score
    "chico": lambda moves: blue2.player(moves, depth=2)[0],  # Returns move, score
    "harpo": lambda moves: blue2.player(moves, depth=3)[0],  # Returns move, score
    "larry": lambda moves: blue1.vector_player(moves, depth=1)[0],  # Returns move, score
    "moe": lambda moves: blue1.vector_player(moves, depth=2)[0],  # Returns move, score
    "curly": lambda moves: blue1.vector_player(moves, depth=3)[0],  # Returns move, score
}

def get_player_names():
    # Marx brothers are broken right now
    return ["groucho", "abbot", "costello"]
    return ["chico", "abbot"]
    return ["chico", "groucho", "harpo"]
    return ["groucho", "abbot", "chico"]
    return ["groucho", "abbot", "larry"]
    return ["curly", "abbot", "groucho"]
    #return ["larry", "abbot"]
    return ["larry", "costello"]
    return ["costello", "harpo"]
    return ["costello", "chico"]
    return ["abbot", "groucho"]
    return all_players.keys()

def get_player(name):

    if name in all_players:
        return all_players[name]

    raise NotImplementedError("The player {} is not implemented in this module".format(name))



