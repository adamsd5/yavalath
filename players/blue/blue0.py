import yavalath_engine
import random
import logging
from enum import Enum
logger = logging.getLogger(__file__)
import pprint

#============================================================================================================
#== Simple minimax player with poor performance
#============================================================================================================

def simple_score_for_option(game_so_far, option):
    result = yavalath_engine.judge_next_move(game_so_far, option)
    score = 0
    if result == yavalath_engine.MoveResult.PLAYER_WINS:
        score = 1000
    if result == yavalath_engine.MoveResult.PLAYER_LOSES:
        score = -1000
    if result == yavalath_engine.MoveResult.ILLEGAL_MOVE:
        score = -5000
    return score


def minimax_score_for_option(board, game_so_far, option, depth, decay=0.9):
    if depth == 0:
        return simple_score_for_option(game_so_far, option)

    # I need to see if I win or lose with this option as-is
    simple_score = simple_score_for_option(game_so_far, option)
    if simple_score != 0:
        return simple_score

    # Consider all responses to your move, and the opponents best score for each
    response_options = set(board.spaces) - set(game_so_far) - set(option)
    response_scored_options = [(minimax_score_for_option(board, game_so_far + [option], o, depth-1), o) for o in response_options]

    # The score for this option for me is the negative of my opponent's best score.
    best_opponent_response = max(response_scored_options)
    return -best_opponent_response[0]*decay


class BoardState:
    def __init__(self, game_so_far):
        self._state = dict()
        self._swap = False
        for turn, move in enumerate(game_so_far):
            if move == "swap":
                self._state[game_so_far[turn-1]] = turn % 2
                self._swap = True
            else:
                self._state[game_so_far[turn-1]] = turn % 2

    def next_state(self, game_so_far):
        assert tuple(game_so_far)

# The GameTracker is an incomplete idea for quickly analysing boards.  I think newer ideas are betteer
class GameTracker:
    class PlayerType(Enum):
        WHITE = 0
        BLACK = 1

    class SpaceTypes(Enum):
        """These types describe a space on the board from the point of view of one of the players.  It might contain a
        stone already, or it might be a winning move, a losing move, a block"""
        WIN = 0
        LOSE = 1
        BLOCK = 2
        STONE = 3


    # Every space has a view of its local world.  It can see in each direction, two spaces.  Those spaces is either
    # white "w", black "b", or empty "-".  When a piece is added, it modifies the view for all 6 neighboring pieces in
    # an easy way.  Each space's view is indexed by the direction.  I can treat off-board spaces the same way.  The
    # view of the world when looking over the edge of the board is always "--" (two empty spaces).

    def __init__(self):
        self.moves = list()
        self.white_moves = list()
        self.black_moves = list()
        self.swapped = False
        self.board = yavalath_engine.HexBoard()
        self.space_to_index = {space:i for i, space in enumerate(self.board.spaces)}
        self.index_to_space = self.board.spaces

        # The views array is indexed first by the space index, then by direction.
        self.views = self.empty_view()

    def empty_view(self):
        return [[list("--") for i in range(6)] for index in range(len(self.index_to_space))]

    def add_move(self, move):
        """I assume white always goes first.  self.moves gets all moves, including the 'swap'.  self.black_moves and
        self.white_moves only get the actual moves where their stones are."""
        if move == "swap":
            assert len(self.moves) == 1, "Can only 'swap' on turn 2"
            assert self.swapped is False, "There was already a swap"
            self.swapped = True
            self.black_moves.append(self.white_moves[0])
            del self.white_moves[0]
            self.moves.append(move)

            # Prepare to update the neighbor weights
            player_type = GameTracker.PlayerType.BLACK
            move = self.black_moves[0]
            # Reset the views.
            self.views = self.empty_view()
        else:
            player = len(self.moves) % 2
            player_type = GameTracker.PlayerType(player)
            if player == 0:
                self.white_moves.append(move)
            else:
                self.black_moves.append(move)
            self.moves.append(move)

        # Update the views
        color = "w" if player == 0 else "b"
        for direction in range(6):
            for distance in range(2):
                next_space = self.board.next_space_in_dir(move, direction, distance+1)
                if next_space is None:
                    continue
                next_space_index = self.space_to_index[next_space]
                reverse_direction = (direction + 3) % 6
                try:
                    self.views[next_space_index][reverse_direction][distance] = color
                except:
                    pprint.pprint(self.views)
                    raise

def player(game_so_far, board=yavalath_engine.HexBoard(), depth=0):
    """This is my attempt at an AI-driven player.  Likely I'll have many variants."""
    # First basic strategy, heading toward minimax.  Score all moves on the current board.
    #return best_move(board, game_so_far, 1)[1]
    options = set(board.spaces) - set(game_so_far)
    #options_with_scores = [(simple_score_for_option(game_so_far, o), o) for o in options]
    options_with_scores = [(minimax_score_for_option(board, game_so_far, o, depth), o) for o in options]
    max_move = max(options_with_scores)
    max_score = max_move[0]
    next_move = random.choice([s for s in options_with_scores if s[0] == max_score])
    logger.debug("Selected move with score of: {}".format(next_move))
    # print("dta Game so far:", game_so_far)
    # print("dta Selected move with score of: {}".format(next_move))
    return next_move[1]
