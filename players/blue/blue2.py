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
from players.blue.classifiers import NextMoveClassifier
import logging
logger = logging.getLogger(__file__)
def is_even(v):
    return v%2 == 0


# Taken from blue1, then modified to use the signature table
class GameState:
    """Each turn, a GameState is created by the player.  This class allows the minimax algo to add and remove moves as
    it is traversing the search tree."""
    def __init__(self, game_so_far, verbose=True):
        self.verbose = verbose
        self.classifier = NextMoveClassifier(game_so_far, verbose=verbose)

        # Populate two matrices, with each column representing the board after a new move is made.  That's how I will
        # evaluate the quality of the move.   The quality of the board after the move.  When a move is selected for
        # further exploration, that move is 'locked in' by setting the correct row of this matrix to 1s or -1s, depending
        # who is considering the move (who's turn it is).  When that move has been fully explored (the recursive call
        # completes), reset the state by setting that row to all zeros again.  The "potential board" column for the
        # move under consideration should't be reset to zeros.
        self.token_for_next_player = 1 if is_even(len(game_so_far)) else -1  # Indicator of who's turn it is.. 1==white, -1==black

        self.compute_conditions()  # Initialize the CxP product matrices

        # My cache.  The thinking cache will be indexed by a pair of sets... the moves white has made and the moves black has made.
        self.thinking_cache = dict()

        # Key spaces is a set of moves that have caused the "best score" to improve during some branch.  It should be
        # first for consideration in moves to make, as it was "important" when considering other moves.
        self.key_spaces = set()
        self.shortcuts = list()

    # I could do all of this functionally by passing in the potential board matrix.  Maybe I'll change it later.
    def add_move(self, move_column_index):
        """
        Puts a move on the 'stack' for exploration.  Updates the black and white potential_moves and
        adds the move to 'game_vec' (which represents the game_so_far).

        Args:
            move_column_index:
                Index in to self.options of the move to add.  This index is only valid within this call from the game
                engine.  This is a valid column index into the potential move matrices, not an index into the game
                board vectors.  (i.e. it is not a row index)
        """
        # lock this move in by setting the correct row of this matrix to 1s or -1s, depending who's turn it is.
        # This makes the move 'actual' for both black and white.
        if self.verbose:
            print("Adding Move:", self.options[move_column_index])

        self.token_for_next_player *= -1

    def undo_move(self, move_column_index):
        """
        Inversion of 'add_move'.  Removes a move from the 'stack'.  This is done when the move has been explored at
        the current recursion depth.  It needs to be put back in the pool for future tree exploration.

        Args:
            move_column_index:
                Index in to self.options of the move to add.  This index is only valid within this call from the game
                engine.  This is a valid column index into the potential move matrices, not an index into the game
                board vectors.  (i.e. it is not a row index)
        """
        # Reverse the actualization of the move from 'add_move' by clearing that space on all potential boards, then
        # setup the potential nature of the move for both matrices in the right cell.
        if self.verbose:
            print("Undoing Move:", self.options[move_column_index])
        self.token_for_next_player *= -1

    def compute_conditions(self):
        self.compute_winning_and_losing_moves()

    def compute_winning_and_losing_moves(self):
        if self.verbose:
            try:
                print("White Wins:{}".format([self.options[i] for i in self.white_winning_moves]))
                print("White Single Checks:{}".format([self.options[i] for i in self.white_single_checks]))
                print("White Multi Checks:{}".format([self.options[i] for i in self.white_multi_checks]))
                print("White Losses:{}".format([self.options[i] for i in self.white_losing_moves]))
                print("Black Wins:{}".format([self.options[i] for i in self.black_winning_moves]))
                print("Black Single Checks:{}".format([self.options[i] for i in self.black_single_checks]))
                print("Black Multi Checks:{}".format([self.options[i] for i in self.black_multi_checks]))
                print("Black Losses:{}".format([self.options[i] for i in self.black_losing_moves]))
            except:
                pass

    def play(self, depth, shortcut_threshold=None):
        """This play routine adds a cacheing layer, turning the search tree into a search graph.  The cache key is
        the moves made by each player (order doesn't matter, but I use tuples for hashability), and the depth.
        Next Steps:
         - Use an int64_t for the move_indices, and set the bits.  Do this if hashing takes a long time.
         - The depth-2 thinking is helpful for the depth-3 move.  How can I use it?"""
        key = (tuple(sorted(self.white_move_indices)), tuple(sorted(self.black_move_indices)), depth)
        if key in self.thinking_cache:
            return self.thinking_cache[key]
        move, score = self.play_uncached(depth)
        self.thinking_cache[key] = (move, score)
        return move, score

    def play_uncached(self, depth, shortcut_threshold=None):
        """
        This is the recursive step for minimax.  Basic idea:
            1. Win if you can win
            2. Block if you can block
            3. Don't make a losing move (3-in-a-row)
            4a. Base case, depth 0: Choose randomly
            4b. Recursive case: Try all potential moves, let the opponent pick his best, and return the worst of those.

        Args:
            depth - How deep for the DFS?  Note that forced moves do not detract from the depth.
            shortcut_threshold - Tells the search to stop if we found a move that is better than this value.  In case
                4b, if I find a move that gives a sufficiently small "opponents best move score", then my response
                score will be high enough that the layer above knows this branch is dead.  In other wrods, if I know I
                can return a move with a score > shortcut_threshold, I can return it right away.  It will elminate the
                layer above from considering the move that allowed me that response.  It doesn't matter if I have even
                better responses.  Likewise, I need to pass in my current "best opponent response" to the next layer to
                tell it to stop looking if it has a strong response.  I'll not consider that move.

        Returns:
             A pair, the index into self.options (also a column index in the potential move matrices) and the score
             for that selected move.
        """
        # I might consider making this functional again so I can more confidently cache/memoize results.
        # Inputs:
        #   * white and black winning and losing moves, which are derived from the 'game_so_far', the board state.
        #   * option_indices, which is derived from the 'game_so_far'.
        #   * The depth
        # So ideally, I would cache a few things... for each game_so_far, have an object that represents my thinking
        # effort for that state.  Given a depth from there, what is my best move and score.  Maybe all allowed branches
        # that go that deep.  Initially, I think there are cases where I go a1, a2, a3, vs. a3, a2, a1.  Both lead to
        # the same game state, and same subsequent thinking.   At depth 3 or more, this happens a lot, enough to keep
        # a simple cache on that.
        if self.verbose:
            print("==== play_uncached({}, {}), token:{}".format(depth, shortcut_threshold, self.token_for_next_player))

        self.compute_winning_and_losing_moves()
        my_wins = self.white_winning_moves if self.token_for_next_player == 1 else self.black_winning_moves
        opp_wins = self.white_winning_moves if self.token_for_next_player == -1 else self.black_winning_moves
        my_checks = self.white_single_checks if self.token_for_next_player == 1 else self.black_single_checks
        opp_checks = self.white_single_checks if self.token_for_next_player == -1 else self.black_single_checks
        my_losses = self.white_losing_moves if self.token_for_next_player == 1 else self.black_losing_moves
        #opp_losses = self.white_losing_moves if self.token_for_next_player == -1 else self.black_losing_moves # Note Used.

        # For now, this is where I filter down to the allowed moves.  Maybe doing it in 'add_move' and 'undo_move' is better
        # Also, this is probably faster as a vector 'and' operation.
        my_wins = my_wins.intersection(self.option_indices)
        opp_wins = opp_wins.intersection(self.option_indices)
        my_losses = my_losses.intersection(self.option_indices)
        my_checks = my_checks.intersection(self.option_indices)  # This should be redundant.
        opp_checks = opp_checks.intersection(self.option_indices)  # This should be redundant.
        #opp_losses = opp_losses.intersection(self.option_indices) # Not used.  Consider whether I should avoid these spaces.

        # Win if you can win
        if len(my_wins):
            if self.verbose:
                print("==== play_uncached({}, {}), token:{}, returns:".format(depth, shortcut_threshold, self.token_for_next_player), list(my_wins)[0], 1000)
            return list(my_wins)[0], 1000

        # Block if you can block without losing (maybe he didn't see it).  Recursion is done to compute the score
        # of the block.  Note that I only avoid loss if at depth=0.  No need to think hard about forced losing paths
        blocks = opp_wins
        if depth == 0:
            blocks = opp_wins - my_losses
        if len(blocks):
            moves_to_consider = blocks
            forced = True
        else:
            non_suicidal_moves = self.option_indices - my_losses
            # If I have nothing but death left, gracefully make a losing move
            if len(non_suicidal_moves) == 0:
                assert len(self.option_indices) > 0, "Somehow we were asked to move with a full board."
                moves_to_consider = self.option_indices
                if self.verbose:
                    print("==== play_uncached({}, {}), token:{}, returns:".format(depth, shortcut_threshold, self.token_for_next_player), list(self.option_indices[0], -1000))
                return list(self.option_indices)[0], -1000
            moves_to_consider = non_suicidal_moves
            forced = False

        if self.verbose:
            print("Blocks:{}, forced:{}, moves_to_consider:{}".format(
                [self.options[b] for b in blocks], forced, [self.options[b] for b in moves_to_consider]))


        # TODO: Any double-check is as good as a win.  Though these would be found with a deeper search, doing it here moves
        # this win condition 2 levels higher.  (1 if I keep diving on forced moves.  Maybe I should keep diving on
        # checks as well.  That would mean more condition vectors, but perhaps that is fine.)

        # Now, pick intelligently from moves_to_consider
        if depth == 0:
            move, score = random.choice(list(moves_to_consider)), 0# TODO: Use better heuristics
            if self.verbose:
                print("==== play_uncached({}, {}), token:{}, returns Random at depth 0:".format(depth, shortcut_threshold, self.token_for_next_player), self.options[move], score)
            return move, score

        # This is the recursive step... find my opponent's best move for each of my possible moves.
        my_moves_giving_worst_opponent_response = list()
        best_opponent_response_score = 2000  # If I don't respond, that is the best for my opponent.
        if shortcut_threshold is None:
            shortcut_threshold = 2000

        # TODO: Sort 'moves_to_consider' based on likelihood to cause shortcuts.
        for i, move_for_consideration_index in enumerate(itertools.chain(moves_to_consider.intersection(self.key_spaces), moves_to_consider-self.key_spaces)):
            move_for_consideration = move_for_consideration_index
            self.add_move(move_for_consideration)

            # If the move is a block, or a check, don't decrease the depth.  This will fully explore the check chains.
            if forced or  move_for_consideration_index in my_checks:
                next_depth = depth
            else:
                next_depth = depth - 1

            # The following is apparently called Alpha-Beta pruning:
            # If I have found a move that gives me a score better than requested, I will shortcut and return that
            # move/score.  Similarly, when asking for the opponent response, I tell the next layer down to shortcut
            # if it finds a response better than what came before.
            opp_move_index, opp_score = self.play(depth=next_depth, shortcut_threshold=best_opponent_response_score)

            if opp_score < best_opponent_response_score:
                self.key_spaces.add(opp_move_index)
                if self.verbose:
                    print("Key space found:", (depth, i, opp_move_index, best_opponent_response_score, opp_score))
                my_moves_giving_worst_opponent_response.clear()
                best_opponent_response_score = opp_score
            if opp_score == best_opponent_response_score:
                my_moves_giving_worst_opponent_response.append(move_for_consideration_index)
                if -opp_score > shortcut_threshold:
                    self.shortcuts.append((depth, i, move_for_consideration_index, -opp_score, shortcut_threshold))
                    break # Stop considering moves... I've found a strong enough response. (Alpha-beta pruning)

            self.undo_move(move_for_consideration)

        # Choose the move that minimizes my opponents best response score.  There may be ties.  Choose randomly among them.
        my_score = -best_opponent_response_score
        if self.verbose:
            verbose_options = [self.options[i] for i in my_moves_giving_worst_opponent_response]
            print("Equally good moves giving score {}:{}".format(my_score, verbose_options))
        selection = random.choice(my_moves_giving_worst_opponent_response)

        if self.verbose:
            print("At depth {}, Selected:".format(depth), self.options[selection])

        my_move_index = selection
        if self.verbose:
            print("==== play_uncached({}, {}), token:{}, returns:".format(depth, shortcut_threshold, self.token_for_next_player), self.options[my_move_index], my_score)
        return my_move_index, my_score


def player(game_so_far, depth=0, verbose=False):
    state = GameState(game_so_far, verbose=verbose)
    move_index, score = state.play(depth)
    result = state.options[move_index]
    #print("Shortcuts:")
    #pprint.pprint(state.shortcuts)
    #print("Key Spaces:")
    #pprint.pprint(state.key_spaces)
    return result, score

