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
from players.blue import blue0, blue1

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

def get_player_names():
    # Marx brothers are broken right now
    return ["abbot", "larry"]
    return ["curly", "abbot", "groucho"]
    #return ["larry", "abbot"]
    return ["larry", "costello"]
    return ["costello", "harpo"]
    return ["costello", "chico"]
    return ["abbot", "groucho"]

def get_player(name):
    if name == "abbot":
        def result_player(moves_so_far):
            return blue0.player(moves_so_far, depth=1)
        return result_player
    if name == "costello":
        def result_player(moves_so_far):
            return blue0.player(moves_so_far, depth=2)
        return result_player
    if name == "larry":
        def result_player(moves_so_far):
            move, score =  blue1.vector_player(moves_so_far, depth=1)
            return move
        return result_player
    if name == "moe":
        def result_player(moves_so_far):
            move, score = blue1.vector_player(moves_so_far, depth=2)
            return move
        return result_player
    if name == "curly":
        def result_player(moves_so_far):
            move, score = blue1.vector_player(moves_so_far, depth=3)
            return move
        return result_player


    raise NotImplementedError("The player {} is not implemented in this module".format(name))



