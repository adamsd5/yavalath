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
from players.blue.classifiers import NextMoveClassifier, SpaceProperies
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
        self.game_so_far = game_so_far
        self.classifier = NextMoveClassifier(game_so_far, verbose=verbose)

        # Populate two matrices, with each column representing the board after a new move is made.  That's how I will
        # evaluate the quality of the move.   The quality of the board after the move.  When a move is selected for
        # further exploration, that move is 'locked in' by setting the correct row of this matrix to 1s or -1s, depending
        # who is considering the move (who's turn it is).  When that move has been fully explored (the recursive call
        # completes), reset the state by setting that row to all zeros again.  The "potential board" column for the
        # move under consideration should't be reset to zeros.
        self.token_for_next_player = 1 if is_even(len(game_so_far)) else -1  # Indicator of who's turn it is.. 1==white, -1==black

        # My cache.  The thinking cache will be indexed by a pair of sets... the moves white has made and the moves black has made.
        self.thinking_cache = dict()
        self.white_move_indices = {self.classifier.space_to_index[move] for move in game_so_far[::2]}
        self.black_move_indices = {self.classifier.space_to_index[move] for move in game_so_far[1::2]}

        # Key spaces is a set of moves that have caused the "best score" to improve during some branch.  It should be
        # first for consideration in moves to make, as it was "important" when considering other moves.
        self.key_spaces = set()
        self.shortcuts = list()

    # I could do all of this functionally by passing in the potential board matrix.  Maybe I'll change it later.
    def add_move(self, option_index):
        if self.verbose:
            print("Adding Move:", self.classifier.options[option_index])
        self.classifier.add_move(option_index, self.token_for_next_player)
        board_index = self.classifier.option_index_to_board_index[option_index]
        if self.token_for_next_player == 1:
            self.white_move_indices.add(board_index)
        else:
            self.black_move_indices.add(board_index)
        self.token_for_next_player *= -1

    def undo_move(self, option_index):
        if self.verbose:
            print("Undoing Move:", self.classifier.options[option_index])
        self.token_for_next_player *= -1
        board_index = self.classifier.option_index_to_board_index[option_index]
        if self.token_for_next_player == 1:  # Note that the player has already changed to what it was at the start of add_move.
            self.white_move_indices.remove(board_index)
        else:
            self.black_move_indices.remove(board_index)
        self.classifier.undo_move(option_index)

    def play(self, depth, shortcut_threshold=None, stack_depth=0, move_stack=None):
        """This play routine adds a cacheing layer, turning the search tree into a search graph.  The cache key is
        the moves made by each player (order doesn't matter, but I use tuples for hashability), and the depth.
        Next Steps:
         - Use an int64_t for the move_indices, and set the bits.  Do this if hashing takes a long time.
         - The depth-2 thinking is helpful for the depth-3 move.  How can I use it?"""
        key = (tuple(sorted(self.white_move_indices)), tuple(sorted(self.black_move_indices)), depth)
        if key in self.thinking_cache:
            return self.thinking_cache[key]
        if move_stack == None:
            move_stack = list()
        move, score = self.play_uncached(depth, shortcut_threshold, stack_depth=stack_depth, move_stack=move_stack)
        self.thinking_cache[key] = (move, score)
        return move, score


    def play_uncached(self, depth, shortcut_threshold=None, stack_depth=0, move_stack=list()):
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
             A pair, the index into self.classifier.options (also a column index in the potential move matrices) and the score
             for that selected move.
        """
        # I might consider making this functional again so I can more confidently cache/memoize results.
        if self.verbose:
            print("==== play_uncached({}, {}), token:{}".format(depth, shortcut_threshold, self.token_for_next_player))

        white_winning_moves = self.classifier.moves_by_property[SpaceProperies.WHITE_WIN]
        black_winning_moves = self.classifier.moves_by_property[SpaceProperies.BLACK_WIN]
        white_losing_moves = self.classifier.moves_by_property[SpaceProperies.WHITE_LOSE]
        black_losing_moves = self.classifier.moves_by_property[SpaceProperies.BLACK_LOSE]
        white_single_checks = self.classifier.moves_by_property[SpaceProperies.WHITE_SINGLE_CHECK]
        black_single_checks = self.classifier.moves_by_property[SpaceProperies.BLACK_SINGLE_CHECK]
        white_double_checks = self.classifier.moves_by_property[SpaceProperies.WHITE_DOUBLE_CHECK]
        black_double_checks = self.classifier.moves_by_property[SpaceProperies.BLACK_DOUBLE_CHECK]

        my_wins = white_winning_moves if self.token_for_next_player == 1 else black_winning_moves
        opp_wins = white_winning_moves if self.token_for_next_player == -1 else black_winning_moves
        my_losses = white_losing_moves if self.token_for_next_player == 1 else black_losing_moves
        opp_losses = white_losing_moves if self.token_for_next_player == -1 else black_losing_moves # Note Used.
        my_checks = white_single_checks if self.token_for_next_player == 1 else black_single_checks
        opp_checks = white_single_checks if self.token_for_next_player == -1 else black_single_checks
        my_double_checks = white_double_checks if self.token_for_next_player == 1 else black_double_checks
        opp_double_checks = white_double_checks if self.token_for_next_player == -1 else black_double_checks

        # NextMoveClassifier makes sure the options are filtered properly... I think.
        # For now, this is where I filter down to the allowed moves.  Maybe doing it in 'add_move' and 'undo_move' is better
        # Also, this is probably faster as a vector 'and' operation.
        my_wins = my_wins.intersection(self.classifier.open_option_indices)
        opp_wins = opp_wins.intersection(self.classifier.open_option_indices)
        my_losses = my_losses.intersection(self.classifier.open_option_indices)
        opp_losses = opp_losses.intersection(self.classifier.open_option_indices) # Not used.  Consider whether I should avoid these spaces.
        my_checks = my_checks.intersection(self.classifier.open_option_indices)  # This should be redundant.
        opp_checks = opp_checks.intersection(self.classifier.open_option_indices)  # This should be redundant.
        my_double_checks = my_double_checks.intersection(self.classifier.open_option_indices)  # This should be redundant.
        opp_double_checks = opp_double_checks.intersection(self.classifier.open_option_indices)  # This should be redundant.

        selected_move = None
        moves_to_consider = None
        forced = False
        force_reason = ""

        # Win if you can win
        if len(my_wins):
            if self.verbose:
                print("==== play_uncached({}, {}), token:{}, returns:".format(depth, shortcut_threshold, self.token_for_next_player), list(my_wins)[0], 1000)
            selected_move = (list(my_wins)[0], 1000)

        # Block if you can block without losing (maybe he didn't see it).  Recursion is done to compute the score
        # of the block.  Note that I only avoid loss if at depth=0.  No need to think hard about forced losing paths
        if selected_move is None:
            blocks = opp_wins - my_losses
            if len(blocks):
                moves_to_consider = blocks
                forced = True
                force_reason = "block"
                if self.verbose:
                    print("Blocks:{}, forced:{}, moves_to_consider:{}".format(
                        [self.classifier.options[b] for b in blocks], forced, [self.classifier.options[b] for b in moves_to_consider]))

        # If we cannot win or block outright, try a double-check
        if selected_move is None and moves_to_consider is None and len(my_double_checks) > 0:
            selected_move = (list(my_double_checks)[0], 999)

        # No outright wins or block, and no double-checks for me, Make a 'double-check' block next.
        if selected_move is None and moves_to_consider is None:
            blocks = opp_double_checks - my_losses
            if len(blocks):
                # TODO: When blocking a double-check, there are 3 choices.  The key space, or either of the win spaces
                # TODO: But, if it is a triple-check, I must take the key space.  For now, take the key space.  I wonder
                # TODO: if the properties table should have the two win spaces in case of double-check.
                moves_to_consider = blocks
                forced = True  # TODO: Not so sure about this one.
                force_reason = "block_dc"
                if self.verbose:
                    print("Double-check Blocks:{}, forced:{}, moves_to_consider:{}".format(
                        [self.classifier.options[b] for b in blocks], forced, [self.classifier.options[b] for b in moves_to_consider]))


        # If all the above selected no moves, try all non-suicidal moves
        if selected_move is None and moves_to_consider is None:
            non_suicidal_moves = self.classifier.open_option_indices - my_losses
            # If I have nothing but death left, gracefully make a losing move
            if len(non_suicidal_moves) == 0:
                assert len(self.classifier.open_option_indices) > 0, "Somehow we were asked to move with a full board."
                if self.verbose:
                    print("==== play_uncached({}, {}), token:{}, returns:".format(depth, shortcut_threshold, self.token_for_next_player), list(self.classifier.open_option_indices[0], -1000))
                selected_move = (list(self.classifier.open_option_indices)[0], -1000)
            else:
                moves_to_consider = non_suicidal_moves
                forced = False

        # =========================================================================================================================================================
        # ========= Perhaps the logic above here and below here should be split.  They connect only with 'selected_move', 'forced', and 'moves_to_consider' =======
        # =========================================================================================================================================================

        # If no move has been selected, choose from among the moves_to_consider
        if selected_move is None:
            assert moves_to_consider is not None and len(moves_to_consider) > 0, "By now, either a move has been selected, or there are some to consider."

            if depth == 0:
                move, score = random.choice(list(moves_to_consider)), 0# TODO: Use better heuristics
                if self.verbose:
                    print("==== play_uncached({}, {}), token:{}, returns Random at depth 0:".format(depth, shortcut_threshold, self.token_for_next_player), self.classifier.options[move], score)
                selected_move = move, score

            else: # depth > 0
                # This is the recursive step... find my opponent's best move for each of my possible moves.
                my_moves_giving_worst_opponent_response = list()
                best_opponent_response_score = 2000  # If I don't respond, that is the best for my opponent.
                if shortcut_threshold is None:
                    shortcut_threshold = 2000

                # TODO: Sort 'moves_to_consider' based on likelihood to cause shortcuts.
                for i, option_index in enumerate(itertools.chain(moves_to_consider.intersection(self.key_spaces), moves_to_consider-self.key_spaces)):
                    self.add_move(option_index)

                    # If the move is a block, or a check, don't decrease the depth.  This will fully explore the check chains.
                    check = option_index in my_checks
                    if forced or check:
                        next_depth = depth
                    else:
                        next_depth = depth - 1

                    # The following is apparently called Alpha-Beta pruning:
                    # If I have found a move that gives me a score better than requested, I will shortcut and return that
                    # move/score.  Similarly, when asking for the opponent response, I tell the next layer down to shortcut
                    # if it finds a response better than what came before.
                    # if len(move_stack) > 20:
                    #     next_depth = 0
                    #     pprint.pprint(("move_stack limit reached:", move_stack, self.classifier.options[option_index]))
                    #     print("game_so_far = {}".format(self.game_so_far))
                    #     raise Exception("Broken")
                    #next_stack = move_stack + [(self.classifier.options[option_index], depth, forced, force_reason, check),]
                    next_stack = move_stack + [self.classifier.options[option_index],]
                    opp_move_index, opp_score = self.play(depth=next_depth, shortcut_threshold=best_opponent_response_score,
                                                          stack_depth=stack_depth+1, move_stack=next_stack)

                    if opp_score < best_opponent_response_score:
                        self.key_spaces.add(opp_move_index)
                        if self.verbose:
                            print("Key space found:", (depth, i, opp_move_index, best_opponent_response_score, opp_score))
                        my_moves_giving_worst_opponent_response.clear()
                        best_opponent_response_score = opp_score
                    if opp_score == best_opponent_response_score:
                        my_moves_giving_worst_opponent_response.append(option_index)

                    self.undo_move(option_index)

                    if -opp_score > shortcut_threshold:  # Negative because for this option, my_score = -opp_score
                        self.shortcuts.append((depth, i, option_index, -opp_score, shortcut_threshold))
                        break # Stop considering moves... I've found a strong enough response. (Alpha-beta pruning)


                # Choose the move that minimizes my opponents best response score.  There may be ties.  Choose randomly among them.
                my_score = -best_opponent_response_score
                if self.verbose:
                    verbose_options = [self.classifier.options[i] for i in my_moves_giving_worst_opponent_response]
                    print("Equally good moves giving score {}:{}".format(my_score, verbose_options))
                my_move_index = random.choice(my_moves_giving_worst_opponent_response)

                if self.verbose:
                    print("At depth {}, Selected:".format(depth), self.classifier.options[my_move_index])

                if self.verbose:
                    print("==== play_uncached({}, {}), token:{}, returns:".format(depth, shortcut_threshold, self.token_for_next_player), self.classifier.options[my_move_index], my_score)

                selected_move = my_move_index, my_score

        assert selected_move is not None, "by now, something must have been selected"
        return selected_move

def player(game_so_far, depth=0, verbose=False):
    state = GameState(game_so_far, verbose=verbose)
    move_index, score = state.play(depth)
    result = state.classifier.options[move_index]
    #print("Shortcuts:")
    #pprint.pprint(state.shortcuts)
    #print("Key Spaces:")
    #pprint.pprint(state.key_spaces)
    return result, score

