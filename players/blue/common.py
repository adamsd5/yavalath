"""Common support routines for the blue player."""
import yavalath_engine
import numpy
import logging
logger = logging.getLogger(__file__)


def moves_to_board_vector(moves):
    board = yavalath_engine.HexBoard()
    space_to_index = {space: i for i, space in enumerate(board.spaces)}
    N = 61+1
    result = numpy.zeros((N,1), dtype="i1")
    for turn, move in enumerate(moves):
        # TODO: Handle 'swap'
        result[space_to_index[move]] = 1 if turn % 2 == 0 else -1
    result[61] = 1 # To account for the level of indicators.
    return result
