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

logger = logging.getLogger("darryl_player")
logger.setLevel(logging.DEBUG)

# Minimax: http://giocc.com/concise-implementation-of-minimax-through-higher-order-functions.html

def get_player_names():
    return ["moe", "abbot"]
    #return ["larry", "abbot"]
    return ["larry", "costello"]
    return ["costello", "harpo"]
    return ["costello", "chico"]
    return ["abbot", "groucho"]

def get_player(name):
    if name == "abbot":
        def result_player(moves_so_far):
            return player(board, moves_so_far, 1)
        return result_player
    if name == "costello":
        def result_player(moves_so_far):
            return player(board, moves_so_far, 2)
        return result_player
    if name == "groucho":
        def result_player(moves_so_far):
            move, score =  vector_player(moves_so_far, depth=1)
            return move
        return result_player
    if name == "chico":
        def result_player(moves_so_far):
            move, score =  vector_player(moves_so_far, depth=2)
            return move
        return result_player
    if name == "harpo":
        def result_player(moves_so_far):
            move, score =  vector_player(moves_so_far, depth=3)
            return move
        return result_player
    if name == "zeppo":
        def result_player(moves_so_far):
            move, score = vector_player(moves_so_far, depth=4)
            return move
        return result_player
    if name == "larry":
        def result_player(moves_so_far):
            move, score =  vector_player2(moves_so_far, depth=3)
            return move
        return result_player
    if name == "moe":
        def result_player(moves_so_far):
            move, score = vector_player2(moves_so_far, depth=2)
            return move
        return result_player
    if name == "curly":
        def result_player(moves_so_far):
            move, score = vector_player2(moves_so_far, depth=1)
            return move
        return result_player


    raise NotImplementedError("The player {} is not implemented in this module".format(name))


#============================================================================================================
# This section is re-usable code for quickly examining board positions
# numpy plan... each board is an Nx1, b vector of 1,0,-1 values.  (For Yavalath, N=61)
# An indicator is a 1xN vector, v such that v*b <= 1.  v*b == 1 implies the condition is met.
# Integer math is faster than floating point math.  So, let v*b == L (level) imply the condition is met.
# Add one more element to b, value 1, and v, value -L.  Then v*b == 0 implies the condition is met.
# Potential moves can then be represented by the boards they create, giving an Nxk matrix, P, representing
# the potential boards.
# All win conditions and loss conditions can be represented in a single M x N matrix, C.  Then, the matrix
# product C*P gives an M*P matrix, telling which conditions hold for each potential board.
# Potentialy, there are multiple M to classify "I Win", "I lose", "I check", "I must block"
#
# For starters, let's get utility functions to translate a set of moves to a board vector and a set of
# condition vectors that determine wins/losses.
#
# Every potential board is the current board plus a move vector for a single move.
#============================================================================================================
board = yavalath_engine.HexBoard()

# ==== The horizon condition vectors approach ===
# IDEA: What if I define a kernel method that applies to the whole board.  For any space, we know whether it is a win,
# loss, check, or block (it could be more than one of these) based on just the 2 spaces in any direction.  Maybe I put
# 3^n on each of those kernel entries, leading to a unique number for each possibility.  There are 12 surrounding
# spaces, for 3^12 values = 531441.  This isn't bad.  Have a lookup array of those values into a condition table.  This
# approach would mean single condition matrix, of 61 condition vectors.  If I do a horizon of 3 spaces, I get a bigger
# lookup table, 3**18 = 387420489, which is big, but not too big for memory.  400MB of my budget.  This would allow me
# to determine checks and multi-checks as well.

# Maybe nos is when I abstract out the board/condition updates from the move selection.  A classified board has this
# interface:
#    __init__(game_so_far)
#    options # list of allowable moves (the output string, like 'e1')
#    column_index_to_board_index  # Maps from the potential move number to the board index of that move
#    add_move(move_index)  # or push_move?
#    undo_move()           # pop_move?
#    # Getters for the list of moves that cause white/black win/loss/check/multicheck (checks need a 3-horizon)
#    classifications  # dict of states -> set of options
#    next_turn_token   #-1 or 1, depending who goes next

NUMBER_OF_BOARD_SPACES = 61
class SpaceProperies(Enum):
    GAME_OVER = "GAME_OVER"
    WHITE_WIN = "WHITE_WIN"
    WHITE_LOSE = "WHITE_LOSE"
    WHITE_SINGLE_CHECK = "WHITE_SINGLE_CHECK"
    WHITE_DOUBLE_CHECK = "WHITE_DOUBLE_CHECK"
    BLACK_WIN = "BLACK_WIN"
    BLACK_LOSE = "BLACK_LOSE"
    BLACK_SINGLE_CHECK = "BLACK_SINGLE_CHECK"
    BLACK_DOUBLE_CHECK = "BLACK_DOUBLE_CHECK"
    
    
class NextMoveClassifier():
    @staticmethod
    @functools.lru_cache()
    def keys_for_token(token):
        if token == 1:
            return SpaceProperies.WHITE_WIN, SpaceProperies.WHITE_LOSE, SpaceProperies.WHITE_DOUBLE_CHECK, SpaceProperies.WHITE_SINGLE_CHECK
        else:
            return SpaceProperies.BLACK_WIN, SpaceProperies.BLACK_LOSE, SpaceProperies.BLACK_DOUBLE_CHECK, SpaceProperies.BLACK_SINGLE_CHECK
            
    @staticmethod
    @functools.lru_cache()
    def find_wins_and_checks_for_token_and_opp_arms(left_arm, right_arm, token):
        """Add to 'properites'.  I plan to call this for each token, for each of the 3 axes.  The properties
        count how many axes have each condition.
        TODO: This needs to improve... really, there is one condition for white and black, either:
           WIN, LOSE, DOUBLE_CHECK, or SINGLE_CHECK, in that order.
        Only one of each color.  That's how to label the space.
        """
        properties = dict()
        win_key, lose_key, double_check_key, single_check_key = NextMoveClassifier.keys_for_token(token)

        # Detect outright wins
        if (left_arm[1] == left_arm[0] == token == right_arm[0]) or (left_arm[0] == token == right_arm[0] == right_arm[1]):
            return win_key

        # And now outright losses
        if (left_arm[1] == left_arm[0] == token) or (left_arm[0] == token == right_arm[0]) or (token == right_arm[0] == right_arm[1]):
            return lose_key

        # And now a double-check on this axis
        if (left_arm[2] == left_arm[1] == right_arm[1] == right_arm[2] == token) and (left_arm[0] == right_arm[0] == 0):
            return double_check_key

        # And now single-checks on this axis, possibly upgrading them to double-checks
        if (left_arm[1] == token and left_arm[0] == 0 and right_arm[0] == token) \
                or (left_arm[0] == token and right_arm[0] == 0 and right_arm[1] == token) \
                or (left_arm[2] == left_arm[1] == token) \
                or (right_arm[2] == right_arm[1] == token):
            return single_check_key

    @staticmethod
    @functools.lru_cache()
    def white_result_from_arm_results(r1, r2, r3):
        arm_results = (r1, r2, r3)
        if SpaceProperies.WHITE_WIN in arm_results:
            white_result = SpaceProperies.WHITE_WIN
        elif SpaceProperies.WHITE_LOSE in arm_results:
            white_result = SpaceProperies.WHITE_LOSE
        elif SpaceProperies.WHITE_DOUBLE_CHECK in arm_results:
            white_result = SpaceProperies.WHITE_DOUBLE_CHECK
        else:
            check_count = arm_results.count(SpaceProperies.WHITE_SINGLE_CHECK)
            if check_count > 1:
                white_result = SpaceProperies.WHITE_DOUBLE_CHECK
            elif check_count == 1:
                white_result = SpaceProperies.WHITE_SINGLE_CHECK
            else:
                white_result = None
        return white_result

    @staticmethod
    @functools.lru_cache()
    def black_result_from_arm_results(r1, r2, r3):
        arm_results = (r1, r2, r3)
        if SpaceProperies.BLACK_WIN in arm_results:
            black_result = SpaceProperies.BLACK_WIN
        elif SpaceProperies.BLACK_LOSE in arm_results:
            black_result = SpaceProperies.BLACK_LOSE
        elif SpaceProperies.BLACK_DOUBLE_CHECK in arm_results:
            black_result = SpaceProperies.BLACK_DOUBLE_CHECK
        else:
            check_count = arm_results.count(SpaceProperies.BLACK_SINGLE_CHECK)
            if check_count > 1:
                black_result = SpaceProperies.BLACK_DOUBLE_CHECK
            elif check_count == 1:
                black_result = SpaceProperies.BLACK_SINGLE_CHECK
            else:
                black_result = None
        return black_result

    @staticmethod
    def signature_to_arms(signature):
        SIGNATURE_OFFSET = sum([3**i for i in range(18)])  # Add this to all signatures to make them >= 0.
        tokens = list()
        for i in range(18):
            modulus = signature % 3
            token = [-1, 0, 1][modulus]
            tokens.append(token)
            signature = int(signature / 3)
        return tuple(tokens[0:3]), tuple(tokens[3:6]), tuple(tokens[6:9]), tuple(tokens[9:12]), tuple(tokens[12:15]), tuple(tokens[15:18])

    @staticmethod
    def compute_signature(arms):
        game_vec = moves_to_board_vector([])[:-1]
        for direction in range(6):
            for distance in range(3):
                token = arms[direction][distance]
                next_space = board.next_space_in_dir('e5', direction=direction, distance=1+distance)
                next_space_index = NextMoveClassifier.space_to_index[next_space]
                game_vec[next_space_index] = token  # -1, 0, or +1
        e5_condition_vector = NextMoveClassifier.get_condition_vector_for_space('e5')
        e5_signature = sum(e5_condition_vector * game_vec)[0]
        return e5_signature

    @staticmethod
    def compute_signature_and_properties(arms):
        """arms is a 6-tuple of the values in the arms pointing out from 'e5'.  Use the 'e5' condition vector and
        compute the signature after setting the peices according to 'arms'.  Then, determine the properties tuple,
        and map the signature to that tuple."""
        assert len(arms) == 6, "Expects 6 arms, but {} were given".format(len(arms))
        game_vec = moves_to_board_vector([])[:-1]
        for direction in range(6):
            for distance in range(3):
                token = arms[direction][distance]
                next_space = board.next_space_in_dir('e5', direction=direction, distance=1+distance)
                assert next_space is not None, "This algo should always be able to find the next space from 'e5'"
                next_space_index = NextMoveClassifier.space_to_index[next_space]
                game_vec[next_space_index] = token  # -1, 0, or +1
        e5_condition_vector = NextMoveClassifier.get_condition_vector_for_space('e5')
        assert len(e5_condition_vector) == len(game_vec), "The condition vector should match the board length, but {} != {}".format(len(e5_condition_vector), len(game_vec))
        e5_signature = sum(e5_condition_vector * game_vec)[0]
        # Now decide what moving at 'e5' does.

        # If any arms are the same pieces, this is GAME_OVER_ALREADY
        result = None
        for arm in arms:
            if arm[0] == arm[1] == arm[2] == 1 or arm[0] == arm[1] == arm[2] == -1:
                result = (SpaceProperies.GAME_OVER,)


        if result is None:
            # Get the white result
            r1 = NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms(arms[0], arms[3], 1)
            r2 = NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms(arms[1], arms[4], 1)
            r3 = NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms(arms[2], arms[5], 1)
            white_result = NextMoveClassifier.white_result_from_arm_results(r1, r2, r3)

            # Get the black result
            r1 = NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms(arms[0], arms[3], -1)
            r2 = NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms(arms[1], arms[4], -1)
            r3 = NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms(arms[2], arms[5], -1)
            black_result = NextMoveClassifier.black_result_from_arm_results(r1, r2, r3)

            result = (white_result, black_result)

        return e5_signature, result


    @staticmethod
    def find_wins_and_checks_for_token_and_opp_arms_slow(properties, left_arm, right_arm, token):
        """Add to 'properites'.  I plan to call this for each token, for each of the 3 axes.  The properties
        count how many axes have each condition.
        TODO: This needs to improve... really, there is one condition for white and black, either:
           WIN, LOSE, DOUBLE_CHECK, or SINGLE_CHECK, in that order.
        Only one of each color.  That's how to label the space.
        """
        lose_key = SpaceProperies.WHITE_LOSE if token == 1 else SpaceProperies.BLACK_LOSE
        win_key = SpaceProperies.WHITE_WIN if token == 1 else SpaceProperies.BLACK_WIN
        single_check_key = SpaceProperies.WHITE_SINGLE_CHECK if token == 1 else SpaceProperies.BLACK_SINGLE_CHECK
        double_check_key = SpaceProperies.WHITE_DOUBLE_CHECK if token == 1 else SpaceProperies.BLACK_DOUBLE_CHECK
        line = list(reversed(left_arm)) + [token,] + list(right_arm)

        # Detect outright wins
        if (line[1] == line[2] == line[3] == line[4]) or (line[2] == line[3] == line[4] == line[5]):
            properties[win_key] = 1
            if lose_key in properties:
                del properties[lose_key]

        # And now outright losses
        elif (line[1] == line[2] == line[3]) or (line[2] == line[3] == line[4]) or (line[3] == line[4] == line[5]):
            if win_key not in properties:
                properties[lose_key] = 1

        # Only consider checks if there are no outright wins or losses
        elif win_key not in properties and lose_key not in properties:
            # And now a double-check on this axis
            if line == [token, token, 0, token, 0, token, token]:
                properties[double_check_key] = 1
                if single_check_key in properties:
                    del properties[single_check_key]

            # And now single-checks on this axis, possibly upgrading them to double-checks
            elif (line[1] == token and line[2] ==0 and line[4] == token) or (line[2] == token and line[4] == 0 and line[5] == token) \
                or (line[0] == line[1] == token) or (line[6] == line[5] == token):
                key = SpaceProperies.WHITE_SINGLE_CHECK if token == 1 else SpaceProperies.BLACK_SINGLE_CHECK
                # If already have a single check, change it to double.  If we already have a double, don't add a single.
                if key in properties:
                    del properties[key]
                    properties[double_check_key] = 1
                elif double_check_key not in properties:
                    properties[key] = 1

    @staticmethod
    @functools.lru_cache()
    def get_condition_vector_for_space(space):
        # Initialize the condition matrix.  It has 61 condition vectors, one for each board space.  There are up to 18
        # non-zero entries in the vector, each a power of 3.
        result = numpy.zeros((NUMBER_OF_BOARD_SPACES, 1), dtype="i4") # TODO: Try performance with i8
        power_of_three = 1
        for direction in range(6):
            for distance in range(3):
                next_space = board.next_space_in_dir(space, direction=direction, distance=distance+1)
                if next_space is not None:  # Check if the horizon space is off the board.
                    next_space_index = NextMoveClassifier.space_to_index[next_space]
                    result[next_space_index] = power_of_three
                power_of_three *= 3  # Always do this, even if the space was off the board
        return result

    @staticmethod
    def compute_signature_and_properties_slow(arms):
        """arms is a 6-tuple of the values in the arms pointing out from 'e5'.  Use the 'e5' condition vector and
        compute the signature after setting the peices according to 'arms'.  Then, determine the properties tuple,
        and map the signature to that tuple."""
        assert len(arms) == 6, "Expects 6 arms, but {} were given".format(len(arms))
        game_vec = moves_to_board_vector([])[:-1]
        e5_condition_vector = NextMoveClassifier.get_condition_vector_for_space('e5')
        for direction in range(6):
            for distance in range(3):
                token = arms[direction][distance]
                next_space = board.next_space_in_dir('e5', direction=direction, distance=1+distance)
                assert next_space is not None, "This algo should always be able to find the next space from 'e5'"
                next_space_index = NextMoveClassifier.space_to_index[next_space]
                game_vec[next_space_index] = token  # -1, 0, or +1
        assert len(e5_condition_vector) == len(game_vec), "The condition vector should match the board length, but {} != {}".format(len(e5_condition_vector), len(game_vec))
        e5_signature = sum(e5_condition_vector * game_vec)[0]
        # Now decide what moving at 'e5' does.

        properties = collections.defaultdict(int)

        # If any arms are the same pieces, this is GAME_OVER_ALREADY
        for arm in arms:
            if arm[0] == arm[1] == arm[2] == 1 or arm[0] == arm[1] == arm[2] == -1:
                properties[SpaceProperies.GAME_OVER] = 1  # The count of GAME_OVER doesn't matter.
        if not SpaceProperies.GAME_OVER in properties:
            for token in [-1, 1]:
                NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms(properties, arms[0], arms[3], token)
                NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms(properties, arms[1], arms[4], token)
                NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms(properties, arms[2], arms[5], token)

        # TODO: Here is the best place to simplify the properties to get one condition for black and one for white.

        return e5_signature, properties

    @staticmethod
    def do_one_piece_of_work(arm1, arm2, arm_possibilities, signature_to_properties_index, all_properties_lists, verbose=False):
        """Iterate over all products where the last of 6 arms is fixed.  I can then parallelize the work based on
        that last arm value."""
        # Each process makes its own tables.  I'll write merge routines later.
        start = time.time()

        for i, arms in enumerate(itertools.product(arm_possibilities, arm_possibilities, arm_possibilities, arm_possibilities)):
            arms = list(arms)
            arms.append(arm1)
            arms.append(arm2)
            signature, properties = NextMoveClassifier.compute_signature_and_properties(arms)
            signature_index = NextMoveClassifier.SIGNATURE_OFFSET + signature
            assert 0 <= signature_index < 2*NextMoveClassifier.SIGNATURE_OFFSET+1, "Invalid signature: {}".format(signature_index)
            if properties not in all_properties_lists:
                signature_to_properties_index[signature_index] = len(all_properties_lists)
                all_properties_lists.append(properties)
            else:
                signature_to_properties_index[signature_index] = all_properties_lists.index(properties)

            if verbose and i % 10000 == 0:
                print("PID:{}, Arms: {}, {}, Done with step {}, duration so far: {}".format(os.getpid(), arm1, arm2, i, time.time() - start))

            # if i > 30000:
            #     break
            #

    @staticmethod
    def worker(worker_id, work_queue, done_queue):
        filename = "data/signature_table_worker_{}.dat".format(worker_id)
        filepath = pathlib.Path(filename)
        if not filepath.exists():
            all_properties_lists = list()  # Indexed by the properties_index that is stored in signature_to_properties_index
            signature_to_properties_index = numpy.zeros(2*NextMoveClassifier.SIGNATURE_OFFSET+1, dtype='i1')
            print("This worker has not done work before.")
        else:
            signature_to_properties_index, all_properties_lists = pickle.load(open(filepath.as_posix(), 'rb'))
            print("Loaded the worker state {}, {}".format(len(signature_to_properties_index), len(all_properties_lists)))

        one_arm_possibilities = list(itertools.product([-1,0,1], [-1,0,1], [-1,0,1]))
        verbose = True or (worker_id == 0)
        try:
            while True:
                task = work_queue.get(timeout=60)
                print("Worker: {}, Starting {} at {} on PID: {}".format(worker_id, task, time.time(), os.getpid()))
                arm1, arm2 = task
                NextMoveClassifier.do_one_piece_of_work(arm1, arm2, one_arm_possibilities, signature_to_properties_index, all_properties_lists, verbose)
                with open(filepath.as_posix(), 'wb') as file_obj:
                    pickle.dump(file=file_obj, obj=(signature_to_properties_index, all_properties_lists))
                    print("Commit to file:{}".format(filename))
                done_queue.put(task)
        except queue.Empty:
            if verbose:
                print("Queue is empty")

    @staticmethod
    def compute_signature_table():
        # I also need to initialize the condition outcomes array.  This has 3**18 entries.  The product of a condition
        # vector and a board vector gives a value in [-N, N] where N = sum([3**i for i in range(18)]) = 193710244
        # Call this number the signature of the space, given the board.
        # (Of course, most of these won't get hit, as it's not possible to have 3 black in a row on one of the arms, for
        # instance.  That would already be a loss for black.)

        done_task_filename = "data/complete_tasks.dat"
        done_task_filepath = pathlib.Path(done_task_filename)
        if not done_task_filepath.exists():
            done_tasks = list()
            print("Nothing done yet.")
        else:
            done_tasks = pickle.load(open(done_task_filepath.as_posix(), 'rb'))
            print("loaded {} done tasks: {}".format(len(done_tasks), done_tasks))

        # Iterate over all possible sets of arms
        work_queue = multiprocessing.Queue()
        done_queue = multiprocessing.Queue()
        one_arm_possibilities = list(itertools.product([-1,0,1], [-1,0,1], [-1,0,1]))
        for arm1, arm2 in itertools.product(one_arm_possibilities, one_arm_possibilities):
            task = (arm1, arm2)
            if task not in done_tasks:
                work_queue.put((arm1, arm2))  # Make 27*27 tasks

        # Make some workers
        processes = [multiprocessing.Process(target=NextMoveClassifier.worker, args=[id, work_queue, done_queue]) for id in range(4)]
        for p in processes:
            p.start()


        while not work_queue.empty():
            try:
                while True:
                    done_task = done_queue.get(timeout=60)
                    done_tasks.append(done_task)
                    print("Waiter... done_task: {}".format(done_task))
                    with open(done_task_filepath.as_posix(), 'wb') as file_obj:
                        pickle.dump(file=file_obj, obj=done_tasks)
            except queue.Empty:
                pass

        for p in processes:
            p.join()

        return True

    space_to_index = {space: i for i, space in enumerate(yavalath_engine.HexBoard().spaces)}
    SIGNATURE_OFFSET = sum([3**i for i in range(18)])  # Add this to all signatures to make them >= 0.

    def __init__(self, game_so_far, verbose=True):
        self.verbose = verbose
        self.options = sorted(list(set(board.spaces) - set(game_so_far)))
        self.option_indices = set(range(len(self.options)))  # Column index into the potential move matrices
        self.game_vec = moves_to_board_vector(game_so_far)

        # Compute the condition matri.  I only care about condition vectors for empty spaces.
        all_condition_vectors = [self.get_condition_vector_for_space(o) for o in self.options]
        matrix_shape = (len(self.options), NUMBER_OF_BOARD_SPACES)
        self.condition_matrix = numpy.matrix(numpy.array(list(itertools.chain(*all_condition_vectors))).reshape(matrix_shape), dtype='i4')
        signature_table, properties_table = NextMoveClassifier.compute_signature_table()

        # Populate the potential moves matrix.  Each column is a board-vector.  I'm not sure I need this.  My space
        # signatures tell me a lot about which moves to make.  The potential move matrices would indeed give me a one
        # move lookahead, but perhaps I can just use the DFS for that.
        # self.column_index_to_board_index = numpy.zeros(shape=len(self.options), dtype='i1')  # Maps potential moves to board locations
        # self.white_potential_moves = numpy.matrix(numpy.zeros(shape=(NUMBER_OF_BOARD_SPACES, len(self.options)), dtype='i1'))
        # self.black_potential_moves = numpy.matrix(numpy.zeros(shape=(NUMBER_OF_BOARD_SPACES, len(self.options)), dtype='i1'))
        # for column_index, option in enumerate(self.options):
        #     self.column_index_to_board_index[column_index] = self.conditions.space_to_index[option]
        #     self.white_potential_moves[:, column_index] = self.game_vec
        #     self.black_potential_moves[:, column_index] = self.game_vec
        #     self.white_potential_moves[self.column_index_to_board_index[column_index], column_index] = 1
        #     self.black_potential_moves[self.column_index_to_board_index[column_index], column_index] = -1

    def compute_winning_and_losing_moves(self):
        """Populate the move sets that will be useful for making decisions on how to move.
        Assumes black and white potential move matrices are current, and the condition_matrix is computed"""
        # white_CxP = self.condition_matrix * self.white_potential_moves
        # black_CxP = self.condition_matrix * self.black_potential_moves
        signatures = self.condition_matrix * self.game_vec # This is "no move"... get signatures for all options
        move_properties = self.signature_table[signatures]

        self.moves_by_property = collections.defaultdict(list)
        for option_index, properties in enumerate(move_properties):
            for property_key, count in properties:
                move_properties[property_key].append(option_index)

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


# ==== The linear condition vectors approach ===
def spaces_to_board_vector(spaces, value):
    space_to_index = {space: i for i, space in enumerate(board.spaces)}
    N = 61+1
    result = numpy.zeros((N,1), dtype="i1")
    for turn, move in enumerate(spaces):
        # TODO: Handle 'swap'
        result[space_to_index[move]] = value
    result[61] = 1 # To account for the level of indicators.
    return result

def spaces_to_condition_vector(spaces, value, level):
    result = spaces_to_board_vector(spaces, value)
    result[61] = level
    return result.transpose()

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



def get_linear_condition_matrix(kernel, delta=0):
    """Applies a linear kernel as much as it can across the board, and returns a condition matrix for this kernel.

    For example, the kernel for winning is (1,1,1,1).  The resulting condition matrix is the set of win conditions.
    The losing kernel is (1,1,1).  The resulting matrix is the set of lose conditions.

    Args:
        kernel:
            Tuple, which will be multiplied by the token values on the board.
        delta:
            By default, the condition is met only if the same piece is at all spaces in the kernel.  For a win, the
            sum must be 4 (or -4) for the condition to be met.  However, for a check (kernel 2,1,1,2), it is useful to
            have the 'condition met' be 5 (or -5), since this can only happen if there are 3 of the same color with
            a space, the check space, between.  In this case, the delta is -1.
    Returns:
        numpy.matrix, with <# conditions> rows and <1 + # board spaces> (= 62) columns.
    """
    """Returns an M x 62 matrix, C, of the win conditions.  Multiply C*P and look for zeros."""
    board = yavalath_engine.HexBoard()
    axis_n_starts = ['e1', 'f1', 'g1', 'h1', 'i1', 'i2', 'i3', 'i4', 'i5']
    axix_0_starts = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 'i1']
    axix_p_starts = ['a5', 'a4', 'a3', 'a2', 'a1', 'b1', 'c1', 'd1', 'e1']
    result = list()
    for coord_u in range(1,10):
        v_width = 9 - abs(coord_u - 5)
        for v_start_pos in range(v_width + 1 - len(kernel)):
            uvK_tuples = list()
            for kernel_offset, kernel_value in enumerate(kernel):
                coord_v = v_start_pos + 1 + kernel_offset
                uvK_tuples.append((coord_u, coord_v, kernel_value))

            # TODO: Should this be global?
            space_to_index = {space: i for i, space in enumerate(yavalath_engine.HexBoard().spaces)}

            def uvK_condition_in_direction(uvK_tuples, axis_starts, axis):
                N = 61 + 1
                one_condition = numpy.zeros((N, 1), dtype="i1")
                for u, v, K in uvK_tuples:
                    space = board.next_space_in_dir(axis_starts[u-1], axis, v-1)
                    board_index = space_to_index[space]
                    one_condition[board_index] = K
                one_condition[N-1] = -sum(kernel) - delta
                return one_condition

            # Add one condition in each of the 3 directions.  The one starting at (coord_u, v_start_pos)
            result.append(uvK_condition_in_direction(uvK_tuples, axix_0_starts, 0)) # Direction 1
            result.append(uvK_condition_in_direction(uvK_tuples, axix_p_starts, 1)) # Direction 2
            result.append(uvK_condition_in_direction(uvK_tuples, axis_n_starts, 5)) # Direction 3

    return numpy.matrix(numpy.array(list(itertools.chain(*result))).reshape((len(result), result[0].shape[0])))


@functools.lru_cache(10)
def get_win_conditions():
    """Returns an M x 62 matrix, C, of the win conditions.  Multiply C*P and look for zeros."""
    # M should be 3*(2 + 3 + 4 + 5 + 6 + 5 + 4 + 3 + 2) = 102
    result = get_linear_condition_matrix((1, 1, 1, 1))
    assert len(result) == 102, "I expect 102 loss conditions"
    return result


@functools.lru_cache(10)
def get_check_conditions():
    """Returns an M x 62 matrix, C, of the win conditions.  Multiply C*P and look for zeros."""
    # M should be 3*(2 + 3 + 4 + 5 + 6 + 5 + 4 + 3 + 2) = 102
    result = get_linear_condition_matrix((2, 1, 1, 2), delta=-1)
    assert len(result) == 102, "I expect 102 check conditions"
    return result


@functools.lru_cache(10)
def get_loss_conditions():
    """Returns an M x 62 matrix, C, of the loss conditions.  Multiply C*P and look for zeros."""
    # M should be 3*(3 + 4 + 5 + 6 + 7 + 6 + 5 + 4 + 3) = 129
    result = get_linear_condition_matrix((1, 1, 1))
    assert len(result) == 129, "I expect 129 loss conditions"
    return result




#============================================================================================================
#== Using numpy arrays, and condition matrices
#============================================================================================================
def vector_minimax_score_for_option(board, game_so_far, option, depth, decay=0.9):
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


class Conditions():
    win_conditions = get_win_conditions()
    loss_conditions = get_loss_conditions()
    check_conditions = get_check_conditions()
    board = yavalath_engine.HexBoard()
    space_to_index = {space:i for i, space in enumerate(board.spaces)}


def is_even(v):
    return v%2 == 0


def vector_player(game_so_far, depth=0, verbose=False):
    """This is my second attmept at an AI-driven player.  Likely I'll have many variants.
    """
    # First basic strategy, heading toward minimax.  Score all moves on the current board.
    # This implementation uses numpy matrices to speed things up
    if verbose:
        print("vector_player({}, {}, {})".format(game_so_far, depth, verbose))
    options = sorted(list(set(board.spaces) - set(game_so_far)))
    if verbose:
        print("Options:", options)
    # TODO: Deal with 'swap'
    # TODO: Make these incremental... they don't change much
    c = Conditions()
    game_vec = moves_to_board_vector(game_so_far)
    logger.debug("Game So Far:", game_vec.transpose())

    # Potential moves can then be represented by the boards they create, giving an Nxk matrix, P, representing
    # the potential boards.  Also compute the opponent's potential moves.  Might be a better way to do this.
    # TODO: potential_moves can be passed to the recursive call, after assigning a candidate move across the entire
    # row for that move's index.  Since the game_so_far will have that move, any matrix operations on the columns that
    # are dead (already moved) will not be considered.
    potential_moves = numpy.zeros(shape=(62, len(options)), dtype='i1')
    opponent_potential_moves = numpy.zeros(shape=(62, len(options)), dtype='i1')
    my_value = 1 if is_even(len(game_so_far)) else -1
    for column, option in enumerate(options):
        potential_moves[:,column] = game_vec
        opponent_potential_moves[:,column] = game_vec
        opponent_potential_moves[c.space_to_index[option], column] = -1*my_value
        potential_moves[c.space_to_index[option], column] = my_value
    potential_moves[-1,:] *= my_value
    opponent_potential_moves[-1,:] *= -1*my_value
    valid_move_indices = set(range(len(options)))

    if verbose:
        print("My Value:", my_value, "Len(game_so_far):", len(game_so_far))
        print("My Potential Moves:")
        pprint.pprint(potential_moves)
        print("Opponent Potential Moves:")
        pprint.pprint(opponent_potential_moves)

    product = c.win_conditions * numpy.matrix(potential_moves)
    winning_move_indices = set()
    for condition_index, move_index in numpy.argwhere(product == 0):
        winning_move_indices.add(move_index)
        if verbose:
            print("Win for move at {}, according to condition {}".format(options[move_index], condition_index))
            print("Board with move:", potential_moves[:,move_index])
            print("Condition vector:", c.win_conditions[condition_index, :])

    product = c.loss_conditions * numpy.matrix(potential_moves)
    losing_move_indices = set()
    for condition_index, move_index in numpy.argwhere(product == 0):
        losing_move_indices.add(move_index)
        if verbose:
            print("Loss for move at {}, according to condition {}".format(options[move_index], condition_index))
            print("Board with move:", potential_moves[:,move_index])
            print("Condition vector:", c.loss_conditions[condition_index, :])

    product = numpy.matrix(c.win_conditions) * numpy.matrix(opponent_potential_moves)
    opp_winning_move_indices = set()
    for condition_index, move_index in numpy.argwhere(product == 0):
        opp_winning_move_indices.add(move_index)
        if verbose:
            print("Opponent win for move at {}, according to condition {}".format(options[move_index], condition_index))
            print("Board with move:", opponent_potential_moves[:, move_index])
            print("Condition vector:", c.win_conditions[condition_index, :])

    product = c.loss_conditions * numpy.matrix(opponent_potential_moves)
    opp_losing_move_indices = set()
    for condition_index, move_index in numpy.argwhere(product == 0):
        opp_losing_move_indices.add(move_index)
        if verbose:
            print("Opponent Loss for move at {}, according to condition {}".format(options[move_index], condition_index))
            print("Board with move:", opponent_potential_moves[:, move_index])
            print("Condition vector:", c.loss_conditions[condition_index, :])

    losing_move_indices -= winning_move_indices
    opp_losing_move_indices -= opp_winning_move_indices
    if verbose:
        print("My Wins:{}".format([options[i] for i in winning_move_indices]))
        print("My Losses:{}".format([options[i] for i in losing_move_indices]))
        print("Opp Wins:{}".format([options[i] for i in opp_winning_move_indices]))
        print("Opp Losses:{}".format([options[i] for i in opp_losing_move_indices]))

    # Win if you can win
    if len(winning_move_indices):
        return options[list(winning_move_indices)[0]], 1000

    # Block if you can block without losing (maybe he didn't see it)
    blocks = opp_winning_move_indices - losing_move_indices
    if len(blocks):
        return options[list(blocks)[0]], 500

    non_suicidal_moves = valid_move_indices - losing_move_indices

    # Now, pick intelligently
    if depth == 0:
        return options[random.choice(list(non_suicidal_moves))], 0  # TODO: Use better heuristics

    opp_responses = list()
    for move_for_consideration_index in non_suicidal_moves:
        move_for_consideration = options[move_for_consideration_index]
        opp_move, opp_score = vector_player(game_so_far + [move_for_consideration,], depth=depth-1, verbose=verbose)  # TODO: If the deeper move is forced, don't decrease depth.
        opp_responses.append((opp_score, move_for_consideration_index, opp_move))
        #pprint.pprint(sorted(opp_responses))
    selection = sorted(opp_responses)[0]   # Return my move that minimizes the opponent's best score.  TODO: Select randomly among ties.
    if verbose:
        print("Selected:", selection)
    opp_score, my_move_index, opp_move = selection
    return options[my_move_index], -opp_score

    # #options_with_scores = [(simple_score_for_option(game_so_far, o), o) for o in options]
    # options_with_scores = [(minimax_score_for_option(game_vec, o, depth), o) for o in options]
    # max_move = max(options_with_scores)
    # max_score = max_move[0]
    # next_move = random.choice([s for s in options_with_scores if s[0] == max_score])
    # logger.debug("Selected move with score of: {}".format(next_move))
    # # print("dta Game so far:", game_so_far)
    # # print("dta Selected move with score of: {}".format(next_move))
    # return next_move[1]
    return None


#============================================================================================================
#== Modify the vector player to try to speed it up.
#============================================================================================================
class GameState:
    """Each turn, a GameState is created by the player.  This class allows the minimax algo to add and remove moves as
    it is traversing the search tree."""
    def __init__(self, game_so_far, verbose=True):
        self.verbose = verbose
        # Allocating memory.  Ideally just do this once per call from the game engine.
        self.options = sorted(list(set(board.spaces) - set(game_so_far)))
        self.option_indices = set(range(len(self.options)))  # Column index into the potential move matrices
        self.game_vec = moves_to_board_vector(game_so_far)
        self.white_potential_moves = numpy.matrix(numpy.zeros(shape=(62, len(self.options)+1), dtype='i1'))
        self.black_potential_moves = numpy.matrix(numpy.zeros(shape=(62, len(self.options)+1), dtype='i1'))
        self.column_index_to_board_index = numpy.zeros(shape=len(self.options), dtype='i1')  # Maps potential moves to board locations
        self.conditions = Conditions()

        # Populate two matrices, with each column representing the board after a new move is made.  That's how I will
        # evaluate the quality of the move.   The quality of the board after the move.  When a move is selected for
        # further exploration, that move is 'locked in' by setting the correct row of this matrix to 1s or -1s, depending
        # who is considering the move (who's turn it is).  When that move has been fully explored (the recursive call
        # completes), reset the state by setting that row to all zeros again.  The "potential board" column for the
        # move under consideration should't be reset to zeros.
        self.token_for_next_player = 1 if is_even(len(game_so_far)) else -1  # Indicator of who's turn it is.. 1==white, -1==black
        for column_index, option in enumerate(self.options):
            self.column_index_to_board_index[column_index] = self.conditions.space_to_index[option]
            self.white_potential_moves[:, column_index] = self.game_vec
            self.black_potential_moves[:, column_index] = self.game_vec
            self.black_potential_moves[self.column_index_to_board_index[column_index], column_index] = -1
            self.white_potential_moves[self.column_index_to_board_index[column_index], column_index] = 1

        # The last column in the potential moves matrix is the board as is, allowing me to detect the changes due to
        # a move.  This is the "no move" column.
        self.white_potential_moves[:, -1] = self.game_vec
        self.black_potential_moves[:, -1] = self.game_vec

        # Set the offset value, to make the condition-vectors work.
        self.white_potential_moves[-1, :] = 1
        self.black_potential_moves[-1, :] = -1
        self.compute_conditions()  # Initialize the CxP product matrices

        # My cache.  The thinking cache will be indexed by a pair of sets... the moves white has made and the moves black has made.
        self.thinking_cache = dict()
        self.white_move_indices = {self.conditions.space_to_index[move] for move in game_so_far[::2]}
        self.black_move_indices = {self.conditions.space_to_index[move] for move in game_so_far[1::2]}

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
        board_index = self.column_index_to_board_index[move_column_index]
        self.white_potential_moves[board_index, :] = self.token_for_next_player
        self.black_potential_moves[board_index, :] = self.token_for_next_player
        self.option_indices.remove(move_column_index)

        if self.token_for_next_player == 1:
            self.white_move_indices.add(board_index)
        else:
            self.black_move_indices.add(board_index)

        # Update the C*P matrices.  Since one value is set to row board_index for all columns in P, that means
        # Column 'board_index' of C is added to all columns of C*P (add the column * token_for_next_player)
        # Note that the column 'move_column_index' for P didn't change since that was the selected move.  But I update
        # it anyhow.  I think that's okay since the move has already been evaluated, and is no longer a valid choice.
        v = self.conditions.win_conditions[:,board_index] * self.token_for_next_player
        self.white_win_CxP += v
        self.black_win_CxP += v
        v = self.conditions.check_conditions[:,board_index] * self.token_for_next_player
        self.white_check_CxP += v
        self.black_check_CxP += v
        v = self.conditions.loss_conditions[:,board_index] * self.token_for_next_player
        self.white_lose_CxP += v
        self.black_lose_CxP += v

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
        board_index = self.column_index_to_board_index[move_column_index]
        self.white_potential_moves[board_index, :] = 0
        self.black_potential_moves[board_index, :] = 0
        self.white_potential_moves[board_index, move_column_index] = 1  # Restore the 'potential' state of this column
        self.black_potential_moves[board_index, move_column_index] = -1 # Restore the 'potential' state of this column
        self.option_indices.add(move_column_index)

        self.token_for_next_player *= -1

        # Undo the stuff from add_move.  Note that it is still -= v.  The token should have changed back to what it was
        # during the CxP updates in add_move by now.
        v = self.conditions.win_conditions[:,board_index] * self.token_for_next_player
        self.white_win_CxP -= v
        self.black_win_CxP -= v
        v = self.conditions.check_conditions[:,board_index] * self.token_for_next_player
        self.white_check_CxP -= v
        self.black_check_CxP -= v
        v = self.conditions.loss_conditions[:,board_index] * self.token_for_next_player
        self.white_lose_CxP -= v
        self.black_lose_CxP -= v

        if self.token_for_next_player == 1:  # Note that the player has already changed to what it was at the start of add_move.
            self.white_move_indices.remove(board_index)
        else:
            self.black_move_indices.remove(board_index)

    def compute_conditions(self):
        # This is probably where the bulk of the work is done.  In each tree node, we compute these four matrix products
        # These are updated incrementally in 'add_move' and 'undo_move'.
        self.white_win_CxP = numpy.matrix(self.conditions.win_conditions) * numpy.matrix(self.white_potential_moves)
        self.black_win_CxP = numpy.matrix(self.conditions.win_conditions) * numpy.matrix(self.black_potential_moves)
        self.white_lose_CxP = numpy.matrix(self.conditions.loss_conditions) * numpy.matrix(self.white_potential_moves)
        self.black_lose_CxP = numpy.matrix(self.conditions.loss_conditions) * numpy.matrix(self.black_potential_moves)
        self.white_check_CxP = numpy.matrix(self.conditions.check_conditions) * numpy.matrix(self.white_potential_moves)
        self.black_check_CxP = numpy.matrix(self.conditions.check_conditions) * numpy.matrix(self.black_potential_moves)
        self.compute_winning_and_losing_moves()

    def compute_winning_and_losing_moves(self):
        self.white_winning_moves = {move_index for condition_index, move_index in numpy.argwhere(self.white_win_CxP == 0)}
        self.black_winning_moves = {move_index for condition_index, move_index in numpy.argwhere(self.black_win_CxP == 0)}
        self.white_losing_moves = {move_index for condition_index, move_index in numpy.argwhere(self.white_lose_CxP == 0)}
        self.black_losing_moves = {move_index for condition_index, move_index in numpy.argwhere(self.black_lose_CxP == 0)}

        # TODO: For checks, I need to count how many checks apply to each space (not potential move).  A double-check is
        # as good as a win.  Problem is that the check condition/kernel doesn't tell which space is the check space.

        # I need to know when each move is adding a new check, not just existing checks.  Same with all conditions, really.
        # I should have a 'no move' option, and subtract condition away.  Add up the columns... that's how many checks
        # for each board.  Subtract away the 'no move' row, and any >0 mean moves that add new checks
        def get_moves_with_new_check_conditions(vec):
            """Note that new check conditions might not be new check spaces."""
            checks = (vec == 0).sum(axis=0)  # Gives a (1,N) row of the sums. N == # potential moves + 1 (the "no move")
            checks[0, :] -= checks[0, -1]

            # There is a problem with checks... It is possible to get a new check condition without getting a new
            # check space.  There are 6 check conditions for the center space.  I need the check spaces, not the
            # conditions.  However, depending on how I use this it is fine.  I will check for win conditions first.
            # I do need to be careful about double-checks... x x - - x -> x x - x x will trigger two check conditions,
            # but it is not a double-check.  I'll just compute single-checks for now.
            new_single_check_moves = {move_index for _, move_index in numpy.argwhere(checks > 0)}
            new_multi_check_moves = set() #{move_index for _, move_index in numpy.argwhere(checks > 1)}
            return new_single_check_moves, new_multi_check_moves

        self.white_single_checks, self.white_multi_checks = get_moves_with_new_check_conditions(self.white_check_CxP)
        self.black_single_checks, self.black_multi_checks = get_moves_with_new_check_conditions(self.black_check_CxP)

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


def vector_player2(game_so_far, depth=0, verbose=False):
    state = GameState(game_so_far, verbose=verbose)
    move_index, score = state.play(depth)
    result = state.options[move_index]
    print("Shortcuts:")
    pprint.pprint(state.shortcuts)
    print("Key Spaces:")
    pprint.pprint(state.key_spaces)
    return result, score

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


def player(board, game_so_far, depth=0):
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


def play_game():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    player1 = player
    player2 = player
    game, result = yavalath_engine.play_two_player_yavalath(player1, player2)
    print(game, result)
    if len(game) > 1:
        renderer = yavalath_engine.Render(yavalath_engine.HexBoard(), game)
        renderer.render_image("game_render.png")
    print(result, "on turn", len(game)-1)

    g = GameTracker()
    for move in game:
        g.add_move(move)
    display = [(g.index_to_space[i], g.views[i]) for i in range(len(g.index_to_space))]
    pprint.pprint(display)


def test_views():
    moves = ['i1', 'c2', 'a2', 'f1', 'g3', 'e5', 'g2', 'c6', 'a1', 'c3', 'b5', 'e3', 'd3', 'i3', 'e1', 'f7', 'i4', 'd1', 'h3', 'i5', 'b2', 'd4']
    g = GameTracker()
    for m in moves:
        g.add_move(m)
        print("Move:", m)
        display = [(g.index_to_space[i], g.views[i]) for i in range(len(g.index_to_space))]
        pprint.pprint(display)

def main():
    play_game()

if __name__ == "__main__":
    main()