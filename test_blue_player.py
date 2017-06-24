import unittest
import yavalath_engine
from players import blue_player
import pprint
import collections
import pickle
import timeit
import numpy

class TestDarrylPlayer(unittest.TestCase):
    def test_moves_to_vec(self):
        print(blue_player.moves_to_board_vector([]))
        print(blue_player.moves_to_board_vector(["e1"]))

    def test_vector_player(self):
        blue_player.vector_player([])


class TestNextMoveClassifier(unittest.TestCase):
    def test_pair_of_arms(self):
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((1, 1, 0), (0, 0, 0), 1), blue_player.SpaceProperies.WHITE_LOSE)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((1, 0, 0), (1, 0, 0), 1), blue_player.SpaceProperies.WHITE_LOSE)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((-1, -1, 0), (0, 0, 0), -1), blue_player.SpaceProperies.BLACK_LOSE)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((-1, 0, 0), (-1, 0, 0), -1), blue_player.SpaceProperies.BLACK_LOSE)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((1, 1, 0), (1, 0, 0), 1), blue_player.SpaceProperies.WHITE_WIN)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((1, 0, 0), (1, 1, 0), 1), blue_player.SpaceProperies.WHITE_WIN)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((-1, -1, 0), (-1, 0, 0), -1), blue_player.SpaceProperies.BLACK_WIN)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((-1, 0, 0), (-1, -1, 0), -1), blue_player.SpaceProperies.BLACK_WIN)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((-1, -1, 0), (-1, 0, 0), 1), None)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((1, -1, 0), (-1, 0, 0), 1), None)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((0, 1, 1), (0, 1, 1), 1), blue_player.SpaceProperies.WHITE_DOUBLE_CHECK)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((0, -1, -1), (0, -1, -1), -1), blue_player.SpaceProperies.BLACK_DOUBLE_CHECK)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((-1, 1, 1), (0, 1, 1), 1), blue_player.SpaceProperies.WHITE_SINGLE_CHECK)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((0, 1, 0), (1, 0, 0), 1), blue_player.SpaceProperies.WHITE_SINGLE_CHECK)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((1, 0, 0), (0, 1, 0), 1), blue_player.SpaceProperies.WHITE_SINGLE_CHECK)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((0, -1, -1), (0, 1, 1), -1), blue_player.SpaceProperies.BLACK_SINGLE_CHECK)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((0, -1, 0), (-1, 0, 0), -1), blue_player.SpaceProperies.BLACK_SINGLE_CHECK)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((-1, 0, 0), (0, -1, 0), -1), blue_player.SpaceProperies.BLACK_SINGLE_CHECK)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((1, 0, 1), (1, 1, 0), 1), blue_player.SpaceProperies.WHITE_WIN)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((-1, 0, 0), (0, -1, 0), -1), blue_player.SpaceProperies.BLACK_SINGLE_CHECK)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((1, 0, 1), (1, 1, 0), 1), blue_player.SpaceProperies.WHITE_WIN)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((-1, 0, 0), (0, -1, 0), -1), blue_player.SpaceProperies.BLACK_SINGLE_CHECK)
        self.assertEqual(blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms((-1, 0, 0), (0, -1, 0), -1), blue_player.SpaceProperies.BLACK_SINGLE_CHECK)

    def test_condition_vector(self):
        space_to_index = {space: i for i, space in enumerate(yavalath_engine.HexBoard().spaces)}
        board = yavalath_engine.HexBoard()
        for space in ['e5', 'e4', 'e6', 'd5', 'f5']:
            condition_vec = blue_player.NextMoveClassifier.get_condition_vector_for_space(space)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 0, 1)], 0], 3**0)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 0, 2)], 0], 3**1)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 0, 3)], 0], 3**2)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 1, 1)], 0], 3**3)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 1, 2)], 0], 3**4)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 1, 3)], 0], 3**5)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 2, 1)], 0], 3**6)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 2, 2)], 0], 3**7)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 2, 3)], 0], 3**8)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 3, 1)], 0], 3**9)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 3, 2)], 0], 3**10)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 3, 3)], 0], 3**11)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 4, 1)], 0], 3**12)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 4, 2)], 0], 3**13)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 4, 3)], 0], 3**14)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 5, 1)], 0], 3**15)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 5, 2)], 0], 3**16)
            self.assertEqual(condition_vec[space_to_index[board.next_space_in_dir(space, 5, 3)], 0], 3**17)

    def test_signature_and_propeties(self):
        signature, properties = blue_player.NextMoveClassifier.compute_signature_and_properties((
            (0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)))
        self.assertEqual(signature, 0)
        self.assertEqual(properties, (None, None))

        signature, properties = blue_player.NextMoveClassifier.compute_signature_and_properties((
            (1,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)))
        self.assertEqual(signature, 1)
        self.assertEqual(properties, (None, None))

        signature, properties = blue_player.NextMoveClassifier.compute_signature_and_properties((
            (0,1,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)))
        self.assertEqual(signature, 3)
        self.assertEqual(properties, (None, None))

        signature, properties = blue_player.NextMoveClassifier.compute_signature_and_properties((
            (1,1,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)))
        self.assertEqual(signature, 4)
        self.assertTupleEqual(properties, (blue_player.SpaceProperies.WHITE_LOSE, None))

        signature, properties = blue_player.NextMoveClassifier.compute_signature_and_properties((
            (-1,-1,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)))
        self.assertEqual(signature, -4)
        self.assertTupleEqual(properties, (None, blue_player.SpaceProperies.BLACK_LOSE))

        signature, properties = blue_player.NextMoveClassifier.compute_signature_and_properties((
            (1,1,0),(0,0,0),(1,1,0),(1,0,0),(0,0,0),(0,0,0)))
        self.assertTupleEqual(properties, (blue_player.SpaceProperies.WHITE_WIN, None))

        signature, properties = blue_player.NextMoveClassifier.compute_signature_and_properties((
            (1,1,0),(1,0,0),(1,1,0),(1,0,0),(1,0,0),(0,0,0)))
        self.assertTupleEqual(properties, (blue_player.SpaceProperies.WHITE_WIN, None))

        signature, properties = blue_player.NextMoveClassifier.compute_signature_and_properties((
            (1,1,0),(1,0,0),(-1,-1,0),(1,0,0),(0,1,0),(0,0,0)))
        self.assertTupleEqual(properties, (blue_player.SpaceProperies.WHITE_WIN, blue_player.SpaceProperies.BLACK_LOSE))

        # TODO: Lots more would be nice, but I think this is enough for now.


class TestConditions(unittest.TestCase):
    def test_win_conditions_board_4(self):
        game_so_far = ['e3', 'b1', 'e4']
        state = blue_player.GameState(game_so_far)
        def index_to_moves(index_iterable):
            return {state.options[index] for index in index_iterable}

        self.assertSetEqual(index_to_moves(state.white_winning_moves), set())
        self.assertSetEqual(index_to_moves(state.black_winning_moves), set())
        self.assertSetEqual(index_to_moves(state.white_losing_moves), {'e2', 'e5'})  # Winning conditions imply 2 losing conditions
        self.assertSetEqual(index_to_moves(state.black_losing_moves),  set())
        self.assertSetEqual(index_to_moves(state.white_single_checks), {'e1', 'e6'})  # This is a new condition, but not a new check.
        self.assertSetEqual(index_to_moves(state.black_single_checks),  set())
        self.assertSetEqual(index_to_moves(state.white_multi_checks), set())
        self.assertSetEqual(index_to_moves(state.black_multi_checks), set())

    def test_win_conditions_board_3(self):
        game_so_far = ['a1', 'b1', 'a2', 'b2', 'a4', 'b4']
        state = blue_player.GameState(game_so_far)
        def index_to_moves(index_iterable):
            return {state.options[index] for index in index_iterable}

        self.assertSetEqual(index_to_moves(state.white_winning_moves), {'a3'})
        self.assertSetEqual(index_to_moves(state.black_winning_moves), {'b3'})
        self.assertSetEqual(index_to_moves(state.white_losing_moves), {'a3'})  # Winning conditions imply 2 losing conditions
        self.assertSetEqual(index_to_moves(state.black_losing_moves), {'b3'})  # Winning conditions imply 2 losing conditions
        self.assertSetEqual(index_to_moves(state.white_single_checks), {'a5'})  # This is a new condition, but not a new check.
        self.assertSetEqual(index_to_moves(state.black_single_checks), {'b5'})
        self.assertSetEqual(index_to_moves(state.white_multi_checks), set())
        self.assertSetEqual(index_to_moves(state.black_multi_checks), set())

    def test_win_conditions_board_2(self):
        game_so_far = ['a1', 'b1', 'a2', 'b2', 'a4']
        state = blue_player.GameState(game_so_far)
        def index_to_moves(index_iterable):
            return {state.options[index] for index in index_iterable}

        self.assertSetEqual(index_to_moves(state.white_winning_moves), {'a3'})
        self.assertSetEqual(index_to_moves(state.black_winning_moves), set())
        self.assertSetEqual(index_to_moves(state.white_losing_moves), {'a3'})  # Winning conditions imply 2 losing conditions
        self.assertSetEqual(index_to_moves(state.black_losing_moves), {'b3'})
        self.assertSetEqual(index_to_moves(state.white_single_checks), {'a5'})  # This is a new condition, but not a new check.
        self.assertSetEqual(index_to_moves(state.black_single_checks), {'b4'})
        self.assertSetEqual(index_to_moves(state.white_multi_checks), set())
        self.assertSetEqual(index_to_moves(state.black_multi_checks), set())

    def test_win_conditions_board_1(self):
        game_so_far = ['a1']
        state = blue_player.GameState(game_so_far)

        def index_to_moves(index_iterable):
            return {state.options[index] for index in index_iterable}

        self.assertSetEqual(index_to_moves(state.white_winning_moves), set())
        self.assertSetEqual(index_to_moves(state.black_winning_moves), set())
        self.assertSetEqual(index_to_moves(state.white_losing_moves), set())
        self.assertSetEqual(index_to_moves(state.black_losing_moves), set())
        self.assertSetEqual(index_to_moves(state.white_single_checks), set())
        self.assertSetEqual(index_to_moves(state.black_single_checks), set())
        self.assertSetEqual(index_to_moves(state.white_multi_checks), set())
        self.assertSetEqual(index_to_moves(state.black_multi_checks), set())

    def test_win_conditions_empty_board(self):
        game_so_far = []
        state = blue_player.GameState(game_so_far)

        def index_to_moves(index_iterable):
            return {state.options[index] for index in index_iterable}

        self.assertSetEqual(index_to_moves(state.white_winning_moves), set())
        self.assertSetEqual(index_to_moves(state.black_winning_moves), set())
        self.assertSetEqual(index_to_moves(state.white_losing_moves), set())
        self.assertSetEqual(index_to_moves(state.black_losing_moves), set())
        self.assertSetEqual(index_to_moves(state.white_single_checks), set())
        self.assertSetEqual(index_to_moves(state.black_single_checks), set())
        self.assertSetEqual(index_to_moves(state.white_multi_checks), set())
        self.assertSetEqual(index_to_moves(state.black_multi_checks), set())

    def test_potential_moves(self):
        def test_game_so_far(game_so_far):
            board = yavalath_engine.HexBoard()
            state = blue_player.GameState(game_so_far)
            self.assertEqual(len(state.options), 61-len(game_so_far), "The number of options is wrong for game: {}.".format(game_so_far))
            options = [space_to_index[space] for space in state.options]
            self.assertEqual(state.white_potential_moves.shape, (62, 62-len(game_so_far))) # 62 rows for the spaces + "1", 61-len(game) columns for potential moves + the "no move"
            for board_space in range(61):  # Always 61, since the board size is fixed
                for potential_move_index in range(len(state.options)):  # Just the number of options
                    expected = 1 if board_space == options[potential_move_index] else 0
                    if board.spaces[board_space] in game_so_far:  # a1
                        expected = 1 if board_space in state.white_move_indices else -1
                    self.assertEqual(state.white_potential_moves[board_space, potential_move_index], expected,
                                     "Mismatch for board_space:{}, potential_move_index:{}".format(board_space, potential_move_index))
            for board_space in range(61):
                potential_move_index = len(state.options)
                expected = 1 if board_space in state.white_move_indices else -1 if board_space in state.black_move_indices else 0
                self.assertEqual(state.white_potential_moves[board_space, potential_move_index], expected,
                                 "Mismatch for 'no move' column, board_space:{}, potential_move_index:{}".format(board_space, potential_move_index))
            for potential_move_index in range(len(state.options)):  # Just the number of options
                board_space = 61
                self.assertEqual(state.white_potential_moves[board_space, potential_move_index], 1,
                                 "Mismatch for 'offset' row, board_space:{}, potential_move_index:{}".format(board_space, potential_move_index))

        def test_all_game_steps(game_so_far):
            for i in range(len(game_so_far)):
                test_game_so_far(game_so_far[0:i])

        # If I think there is a gam ewith the potential moves being incorrectly created, add that game here.
        test_all_game_steps(['a1', 'b2', 'a2', 'b1', 'c3'])
        game_so_far = ['g1', 'd5', 'e7', 'b5', 'e1', 'e3', 'i4', 'g4', 'b6', 'e4', 'f5', 'a5', 'c5', 'e8', 'c3', 'd3', 'f4', 'd2']
        test_all_game_steps(game_so_far)


def main1():
    win_C = blue_player.get_win_conditions()
    loss_C = blue_player.get_loss_conditions()
    pprint.pprint(win_C)
    pprint.pprint(win_C.shape)
    pprint.pprint(loss_C)
    pprint.pprint(loss_C.shape)
    print(blue_player.moves_to_board_vector(["e1", "e2", "e3", "e4"]).transpose())


class TestMoves(unittest.TestCase):

    def test_expected(self):
        player = lambda game: blue_player.vector_player2(game, depth=2)
        blocks = [
            # Pairs of game_so_far, expected_move
            (['a1', 'b1', 'a2', 'b2', 'a4'], 'a3'),
            (['a1', 'b1', 'a2', 'b2', 'a4', 'b4'], 'a3'),
            ]
        for game_so_far, expected_block in blocks:
            move, score = player(game_so_far)
            self.assertEqual(move, expected_block)




def main2():
    import logging
    blue_player.logger.setLevel(logging.DEBUG)
    game_so_far = ['g1', 'd5', 'e7', 'b5', 'e1', 'e3', 'i4', 'g4', 'b6', 'e4', 'f5', 'a5', 'c5', 'e8', 'c3', 'd3', 'f4', 'd2']
    #game_so_far =  ['e8', 'a1', 'a2', 'a3', 'f1', 'a4', 'c1']
    #game_so_far =  ['d3']
    print("Starting white spaces:", sorted(game_so_far[::2]))
    print("Starting black spaces:", sorted(game_so_far[1::2]))

    r = blue_player.vector_player2(game_so_far, depth=2, verbose=True)
    print("Result:", r)

def main3():
    v1 = blue_player.get_linear_condition_matrix((1, 1, 1, 1))
    v2 = blue_player.get_win_conditions()
    print("v1 == v2:", (v1 == v2).all())


    v1 = blue_player.get_linear_condition_matrix((1, 1, 1))
    v2 = blue_player.get_loss_conditions()
    print("v1 == v2:", (v1 == v2).all())


space_to_index = {space: i for i, space in enumerate(yavalath_engine.HexBoard().spaces)}


def get_board_hash(white_move_indices, black_move_indices):
    key = (tuple(sorted(white_move_indices)), tuple(sorted(black_move_indices)))
    return hash(key)

def get_board_hash2(white_move_indices, black_move_indices):
    white_bitset = int()
    for move in white_move_indices:
        white_bitset |= (1 << move)
    black_bitset = int()
    for move in black_move_indices:
        black_bitset |= (1 << move)
    key = (white_bitset, black_bitset)
    return hash(key)

def main4():
    game_so_far = "g1 d5 e7 b5 e1 e3 i4 g4 b6 e4 f5 a5 c5 e8 c3 d3 f4 d2 d4".split()
    print("Starting white spaces:", game_so_far[::2])
    print("Starting black spaces:", game_so_far[1::2])

    white_move_indices = {space_to_index[move] for move in game_so_far[::2]}
    black_move_indices = {space_to_index[move] for move in game_so_far[1::2]}

    t = timeit.timeit(lambda: get_board_hash(white_move_indices, black_move_indices))
    print("get_board_hash: 1M iterations took {}s".format(t))

    t = timeit.timeit(lambda: get_board_hash2(white_move_indices, black_move_indices))
    print("get_board_hash2: 1M iterations took {}s".format(t))

def to_str(p):
    if len(p) == 1:
        return p[0].value
    else:
        white, black = p
        return "{}, {}".format(str(white) if white is None else white.value, str(black) if black is None else black.value)

def main5():
    signature_table, properties_table = pickle.load(open("test.dat", 'rb'))

    for p in sorted([to_str(s) for s in properties_table]):
        print(p)
    print("Done")

def main6():
    properties = dict()
    t = timeit.timeit(lambda: blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms_fast((-1, 0, 0), (0, -1, 0), -1))
    print(t)

def main7():
    done_tasks = pickle.load(open("data/backup/complete_tasks.dat", 'rb'))
    SIGNATURE_OFFSET = sum([3**i for i in range(18)])  # Add this to all signatures to make them >= 0.


    for i in [0, 1, 2, 3]:
        filename = "data/backup/signature_table_worker_{}.dat".format(i)
        print("File:{}".format(filename))
        signature_table, properties_table = pickle.load(open(filename, 'rb'))
        #full_table = properties_table[signature_table]
        #print(full_table)
        print("Max:", signature_table.max())
        print(len(properties_table))
        # signature = numpy.where(signature_table == 26)[0]
        # arms = blue_player.NextMoveClassifier.signature_to_arms(int(signature))
        # signature, properties = blue_player.NextMoveClassifier.compute_signature_and_properties(arms)
        # print(signature+SIGNATURE_OFFSET, properties)
        # print(properties_table)
        #
        # for p in sorted([to_str(s) for s in properties_table]):
        #     print(p)

def main8():
    signatures = [
            303081869,
            306723378,
            44801543,
            173941706,
        ]
    for signature in signatures:
        SIGNATURE_OFFSET = sum([3**i for i in range(18)])  # Add this to all signatures to make them >= 0.
        arms = blue_player.NextMoveClassifier.signature_to_arms(int(signature))
        print(signature, arms, SIGNATURE_OFFSET+blue_player.NextMoveClassifier.compute_signature(arms))
        signature, properties = blue_player.NextMoveClassifier.compute_signature_and_properties(arms)
        print(signature+SIGNATURE_OFFSET, properties)

if __name__ == "__main__":
    main7()
   # blue_player.NextMoveClassifier.compute_signature_table()