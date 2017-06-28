import yavalath_engine
from players.blue import common

from enum import Enum
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
import logging
logger = logging.getLogger(__file__)

# TODO: More classification should allow for more strategies.  TRIPLE_CHECK, for instance means there is only one space
# allowed for blocking.  (You can prevent a DOUBLE_CHECK playing in 3 different spaces.)  It would be nice to include
# in the properties the locations of the checks.  It is either arm[0], arm[1], or neither for the 6 arms, or 3^6 = 729 
# possibilities.  That would make the properties table bigger than 256, so I'd need 2 bytes per signature.  That would
# double the signature table size, but it would also open up more possibilities.  On second thought, I would need more.
# I'd need checks for each color.  Just adding TRIPLE_CHECK would mean 37 entries in the properties table.
# It might also be nice to know how many triangles are formed.  
NUMBER_OF_BOARD_SPACES = 61
class SpaceProperies(Enum):
    GAME_OVER = "GAME_OVER"
    WHITE_WIN = "WHITE_WIN"
    WHITE_LOSE = "WHITE_LOSE"
    WHITE_SINGLE_CHECK = "WHITE_SINGLE_CHECK"
    WHITE_DOUBLE_CHECK = "WHITE_DOUBLE_CHECK"
    WHITE_TRIPLE_CHECK = "WHITE_TRIPLE_CHECK"
    BLACK_WIN = "BLACK_WIN"
    BLACK_LOSE = "BLACK_LOSE"
    BLACK_SINGLE_CHECK = "BLACK_SINGLE_CHECK"
    BLACK_DOUBLE_CHECK = "BLACK_DOUBLE_CHECK"
    BLACK_TRIPLE_CHECK = "BLACK_TRIPLE_CHECK"


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
        line = (left_arm[2], left_arm[1], left_arm[0], token, right_arm[0], right_arm[1], right_arm[2])
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

        if (left_arm[1] == token == right_arm[0] == right_arm[2]) and (left_arm[0] == right_arm[1] == 0):
            return double_check_key

        if (left_arm[2] == left_arm[0] == token == right_arm[1]) and (left_arm[1] == right_arm[0] == 0):
            return double_check_key

        # And now single-checks on this axis, possibly upgrading them to double-checks
        if (left_arm[1] == token and left_arm[0] == 0 and right_arm[0] == token) \
                or (left_arm[0] == token and right_arm[0] == 0 and right_arm[1] == token) \
                or (left_arm[2] == left_arm[1] == token and left_arm[0] == 0) \
                or (left_arm[2] == left_arm[0] == token and left_arm[1] == 0) \
                or (right_arm[2] == right_arm[1] == token and right_arm[0] == 0) \
                or (right_arm[2] == right_arm[0] == token and right_arm[1] == 0):
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
            double_check_count = arm_results.count(SpaceProperies.WHITE_DOUBLE_CHECK)
            total_check_count = 2*double_check_count + check_count
            
            if total_check_count > 2:
                white_result = SpaceProperies.WHITE_TRIPLE_CHECK
            elif total_check_count > 1:
                white_result = SpaceProperies.WHITE_DOUBLE_CHECK
            elif total_check_count == 1:
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
            double_check_count = arm_results.count(SpaceProperies.BLACK_DOUBLE_CHECK)
            total_check_count = 2*double_check_count + check_count
            
            if total_check_count > 2:
                black_result = SpaceProperies.BLACK_TRIPLE_CHECK
            elif total_check_count > 1:
                black_result = SpaceProperies.BLACK_DOUBLE_CHECK
            elif total_check_count == 1:
                black_result = SpaceProperies.BLACK_SINGLE_CHECK
            else:
                black_result = None
        return black_result

    @staticmethod
    def signature_index_to_arms(signature):
        tokens = list()
        for i in range(18):
            modulus = signature % 3
            token = [-1, 0, 1][modulus]
            tokens.append(token)
            signature = int(signature / 3)
        return tuple(tokens[0:3]), tuple(tokens[3:6]), tuple(tokens[6:9]), tuple(tokens[9:12]), tuple(tokens[12:15]), tuple(tokens[15:18])

    @staticmethod
    def compute_signature(arms):
        game_vec = common.moves_to_board_vector([])[:-1]
        for direction in range(6):
            for distance in range(3):
                token = arms[direction][distance]
                next_space = yavalath_engine.HexBoard().next_space_in_dir('e5', direction=direction, distance=1+distance)
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
        game_vec = common.moves_to_board_vector([])[:-1]
        for direction in range(6):
            for distance in range(3):
                token = arms[direction][distance]
                next_space = yavalath_engine.HexBoard().next_space_in_dir('e5', direction=direction, distance=1+distance)
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
                next_space = yavalath_engine.HexBoard().next_space_in_dir(space, direction=direction, distance=distance+1)
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
        game_vec = common.moves_to_board_vector([])[:-1]
        e5_condition_vector = NextMoveClassifier.get_condition_vector_for_space('e5')
        for direction in range(6):
            for distance in range(3):
                token = arms[direction][distance]
                next_space = yavalath_engine.HexBoard().next_space_in_dir('e5', direction=direction, distance=1+distance)
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
    def do_one_piece_of_work(arm1, arm2, arm_possibilities, signature_to_properties_index, properties_table, verbose=False):
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
            # assert 0 <= signature_index < 2*NextMoveClassifier.SIGNATURE_OFFSET+1, "Invalid signature: {}".format(signature_index)
            # assert properties in properties_table, "Unforutnately, {} is not in the properties_table".format(properties)
            signature_to_properties_index[signature_index] = properties_table.index(properties)
            if verbose and i % 10000 == 0:
                print("PID:{}, Arms: {}, {}, Done with step {}, duration so far: {}".format(os.getpid(), arm1, arm2, i, time.time() - start))

    @staticmethod
    @functools.lru_cache()
    def all_valid_properties():
        """This returns the list of all expected tuples that can come out of the compute_signature_and_properties function.
        """
        valid_black_properties = [None, SpaceProperies.BLACK_WIN, SpaceProperies.BLACK_LOSE, SpaceProperies.BLACK_SINGLE_CHECK, SpaceProperies.BLACK_DOUBLE_CHECK]
        valid_white_properties = [None, SpaceProperies.WHITE_WIN, SpaceProperies.WHITE_LOSE, SpaceProperies.WHITE_SINGLE_CHECK, SpaceProperies.WHITE_DOUBLE_CHECK]
        result = [(SpaceProperies.GAME_OVER,)] + list(itertools.product(valid_white_properties, valid_black_properties))
        return result

    @staticmethod
    def worker(worker_id, work_queue, done_queue):
        filename = "data/signature_table_worker_{}.dat".format(worker_id)
        filepath = pathlib.Path(filename)
        properties_table = NextMoveClassifier.all_valid_properties()
        if not filepath.exists():
            signature_to_properties_index = numpy.zeros(2*NextMoveClassifier.SIGNATURE_OFFSET+1, dtype='i1') - 1 # Initialize everything to sentinel, -1
            print("This worker has not done work before.")
        else:
            signature_to_properties_index, loaded_properties_table = pickle.load(open(filepath.as_posix(), 'rb'))
            assert loaded_properties_table == properties_table
            print("Loaded the worker state {}, {}".format(len(signature_to_properties_index), len(properties_table)))

        one_arm_possibilities = list(itertools.product([-1,0,1], [-1,0,1], [-1,0,1]))
        verbose = True or (worker_id == 0)
        try:
            while True:
                task = work_queue.get(timeout=60)
                print("Worker: {}, Starting {} at {} on PID: {}".format(worker_id, task, time.time(), os.getpid()))
                arm1, arm2 = task
                NextMoveClassifier.do_one_piece_of_work(arm1, arm2, one_arm_possibilities, signature_to_properties_index, properties_table, verbose)
                with open(filepath.as_posix(), 'wb') as file_obj:
                    pickle.dump(file=file_obj, obj=(signature_to_properties_index, properties_table))
                    print("Commit to file:{}".format(filename))
                done_queue.put(task)
        except queue.Empty:
            if verbose:
                print("Queue is empty")

    @staticmethod
    def compute_signature_table(tasks=None, processes=4):
        """Starts a multi-processed computation of the signature/properties tables.  Data will be put in 'data/'.  If
        'tasks' is passed in as an iterable of pairs of arms, that will be used instead of the full desired set of
        27 x 27 arm paris."""
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
        if tasks is None:
            one_arm_possibilities = list(itertools.product([-1,0,1], [-1,0,1], [-1,0,1]))
            for arm1, arm2 in itertools.product(one_arm_possibilities, one_arm_possibilities):
                task = (arm1, arm2)
                if task not in done_tasks:
                    work_queue.put((arm1, arm2))  # Make 27*27 tasks
        else:
            for task in tasks:
                work_queue.put(task)

        # Make some workers
        processes = [multiprocessing.Process(target=NextMoveClassifier.worker, args=[id, work_queue, done_queue]) for id in range(processes)]
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

        # One more pass to mark things as done
        try:
            while True:
                done_task = done_queue.get(timeout=1)
                done_tasks.append(done_task)
                print("Waiter... done_task: {}".format(done_task))
                with open(done_task_filepath.as_posix(), 'wb') as file_obj:
                    pickle.dump(file=file_obj, obj=done_tasks)
        except queue.Empty:
            pass

        return True

    @staticmethod
    @functools.lru_cache()
    def load_signature_table():
        filename = pathlib.Path(__file__).parent / "signature_table.dat"
        print("Loading signature table from: {}.  Exists: {}".format(filename.as_posix(), filename.exists()))
        return pickle.load(open(filename.as_posix(), "rb"))

    @staticmethod
    def get_board_properties(game_so_far, verbose=False):
        c = NextMoveClassifier(game_so_far, verbose=verbose)
        c.compute_moves_by_property()
        return c.moves_by_property

    space_to_index = {space: i for i, space in enumerate(yavalath_engine.HexBoard().spaces)}
    SIGNATURE_OFFSET = sum([3**i for i in range(18)])  # Add this to all signatures to make them >= 0.

    def __init__(self, game_so_far, verbose=True):
        self.verbose = verbose
        self.options = sorted(list(set(yavalath_engine.HexBoard().spaces) - set(game_so_far)))
        self.option_index_to_board_index = [NextMoveClassifier.space_to_index[s] for s in self.options]
        self.open_option_indices = set(range(len(self.options)))  # Column index into the potential move matrices
        self.game_vec = common.moves_to_board_vector(game_so_far)[:NUMBER_OF_BOARD_SPACES]

        # Compute the condition matrix.  I only care about condition vectors for empty spaces.
        all_condition_vectors = [self.get_condition_vector_for_space(o) for o in self.options]
        matrix_shape = (len(self.options), NUMBER_OF_BOARD_SPACES)
        self.condition_matrix = numpy.matrix(numpy.array(list(itertools.chain(*all_condition_vectors))).reshape(matrix_shape), dtype='i4')

        # Initialize the signatures now.  Manage them with add_move/undo_move.
        self.signatures = self.condition_matrix * self.game_vec

        # Takes days to compute the table
        self.signature_table, self.properties_table = NextMoveClassifier.load_signature_table()
        self.properties_table = numpy.array(self.properties_table)
        self.signature_table = numpy.array(self.signature_table)

        self.compute_moves_by_property()

    def compute_moves_by_property(self):
        """Populate the move sets that will be useful for making decisions on how to move.
        Assumes black and white potential move matrices are current, and the condition_matrix is computed"""
        move_properties = self.properties_table[self.signature_table[self.signatures + NextMoveClassifier.SIGNATURE_OFFSET]]

        self.moves_by_property = collections.defaultdict(set)
        for option_index, properties in enumerate(move_properties):
            for property_key in properties:
                for sub_key in property_key:
                    self.moves_by_property[sub_key].add(option_index)

        if self.verbose:
            print("Moves by property:")
            for key, option_index_list in self.moves_by_property.items():
                if key is None:
                    continue
                option_list = [self.options[i] for i in option_index_list]
                print("{}: {}".format(key, sorted(option_list)))

    def add_move(self, option_index, token):
        board_index = self.option_index_to_board_index[option_index]
        # assert self.game_vec[board_index,0] == 0, "Attempting to add a move to a non-empty space: {}, {}".format(self.game_vec, board_index)
        # assert option_index in self.open_option_indices, "Attept to add a move that is not available."
        self.game_vec[board_index] = token
        self.open_option_indices.remove(option_index)

        # This is the logical: self.signatures = self.condition_matrix * self.game_vec
        # Optimized, I only need to change by one row in the condition matrix
        self.signatures += self.condition_matrix[:, board_index] * token  # Add the correct column, multiplied by the new token
        # if numpy.max(self.signatures) + NextMoveClassifier.SIGNATURE_OFFSET >= 387420489:
        #     raise Exception("Broken")
        self.compute_moves_by_property()

    def undo_move(self, option_index):
        board_index = self.option_index_to_board_index[option_index]
        # assert option_index not in self.open_option_indices, "Attept to add a undo a move that is already available."
        self.open_option_indices.add(option_index)
        token = self.game_vec[board_index][0]
        # assert token in (-1, 1), "Attempting to remove a move from an empty space: {}, {}".format(self.game_vec, board_index)
        self.game_vec[board_index] = 0
        self.signatures -= self.condition_matrix[:, board_index] * token
        # if numpy.max(self.signatures) + NextMoveClassifier.SIGNATURE_OFFSET >= 387420489:
        #     raise Exception("Broken")
        self.compute_moves_by_property()

    def compute_signature_and_properties_for_space(self, space):
        """Uses the current board and target space to return the signature for that space, and the computed (not cached)
        properties for that space.  This is for testing."""
        arms = list()
        board = yavalath_engine.HexBoard()
        space_to_index = {space: i for i, space in enumerate(board.spaces)}
        arms = list()
        for direction in range(6):
            arm = list()
            for distance in range(3):
                next_space = board.next_space_in_dir(space, direction, distance+1)
                if next_space is None:
                    token = 0
                else:
                    next_space_index = space_to_index[next_space]
                    token = self.game_vec[next_space_index][0]
                arm.append(token)
            arms.append(tuple(arm))
        return NextMoveClassifier.compute_signature_and_properties(tuple(arms))
