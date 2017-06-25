import unittest
import yavalath_engine
from players import blue_player
import pprint
import collections
import pickle
import timeit
import numpy
import pathlib
import itertools

class TestDarrylPlayer(unittest.TestCase):
    def test_moves_to_vec(self):
        print(blue_player.moves_to_board_vector([]))
        print(blue_player.moves_to_board_vector(["e1"]))

    def test_vector_player(self):
        blue_player.vector_player([])


class TestNextMoveClassifier(unittest.TestCase):
    def setUp(self):
        self.signature_table, self.properties_table = pickle.load(open("signature_tables.dat", "rb"))
        self.properties_table = numpy.array(self.properties_table)
        self.signature_table = numpy.array(self.signature_table)

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

    def test_signature_lookup_table(self):
        test_signatures = numpy.array([129140163, 43046721, 14348907])  # Various powers of 3 should not have any properties
        move_properties = self.properties_table[self.signature_table[test_signatures]]
        pprint.pprint( sorted(test_signatures[numpy.nonzero(test_signatures)].flatten().tolist()) )
        pprint.pprint(self.signature_table[test_signatures])

    def test_adhoc_classifications(self):
        from players.blue_player import SpaceProperies
        game_so_far = ['g1', 'd5', 'e7', 'b5', 'e1', 'e3', 'i4', 'g4', 'b6', 'e4', 'f5', 'a5', 'c5', 'e8', 'c3', 'd3', 'f4', 'd2']
        # Moves by property:
        # SpaceProperies.WHITE_LOSE: ['c4', 'd6', 'f1', 'f3', 'f6']
        # ERROR: SpaceProperies.BLACK_SINGLE_CHECK: ['b1', 'b3', 'c6', 'd7', 'e6', 'g3'] # b3 is not a check... it is already blocked.
        # SpaceProperies.BLACK_WIN: ['d4']
        # SpaceProperies.BLACK_LOSE: ['c1', 'c2', 'd1', 'e2', 'e5', 'f2', 'f3']
        # ERROR: SpaceProperies.GAME_OVER: ['f7', 'h1']
        # SpaceProperies.WHITE_SINGLE_CHECK: ['d4', 'e5', 'f2']
        classifier = blue_player.NextMoveClassifier(game_so_far=game_so_far)
        classifier.compute_winning_and_losing_moves()
        gamestate = blue_player.GameState(game_so_far=game_so_far)
        self.assertSetEqual(set(gamestate.white_winning_moves), classifier.moves_by_property[blue_player.SpaceProperies.WHITE_WIN])
        self.assertSetEqual(set(gamestate.black_winning_moves), classifier.moves_by_property[blue_player.SpaceProperies.BLACK_WIN])
        signature, properties = classifier.compute_signature_and_properties_for_space('b3')
        self.assertTupleEqual((None,None), properties)
        signature, properties = classifier.compute_signature_and_properties_for_space('f7')
        self.assertTupleEqual((SpaceProperies.WHITE_SINGLE_CHECK,None), properties)
        signature, properties = classifier.compute_signature_and_properties_for_space('f2')
        self.assertTupleEqual((SpaceProperies.WHITE_SINGLE_CHECK,SpaceProperies.BLACK_LOSE), properties)

    def test_signature_lookup_table_with_random_signatures(self):
        SIGNATURE_OFFSET = sum([3**i for i in range(18)])  # Add this to all signatures to make them >= 0.
        signature_index = 129140163 + SIGNATURE_OFFSET  # This is the
        arms = blue_player.NextMoveClassifier.signature_index_to_arms(signature_index)
        #arms = (0,1,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)
        new_signature, properties = blue_player.NextMoveClassifier.compute_signature_and_properties(arms) # Compute slowly
        self.assertEqual(new_signature, signature_index - SIGNATURE_OFFSET, "The signatures should be the same.")
        move_properties = self.properties_table[self.signature_table[signature_index]]
        self.assertEqual(properties, move_properties)


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
    signature_table, properties_table = pickle.load(open("final.dat", 'rb'))

    for p in sorted([to_str(s) for s in properties_table]):
        print(p)
    index = numpy.nonzero(signature_table)
    print("Number of non-zero:", len(index[0]))
    print("Done")

def main6():
    properties = dict()
    t = timeit.timeit(lambda: blue_player.NextMoveClassifier.find_wins_and_checks_for_token_and_opp_arms_fast((-1, 0, 0), (0, -1, 0), -1))
    print(t)

def combine_results(dirs = [r"data\laptop_files", r"data\desktop_files"]):
    final_complete_tasks_filename = "final_complete_tasks.dat"
    final_tables_filename = "final_tables.dat"


    one_arm_possibilities = list(itertools.product([-1,0,1], [-1,0,1], [-1,0,1]))
    desired_tasks = set()
    for arm1, arm2 in itertools.product(one_arm_possibilities, one_arm_possibilities):
        task = (arm1, arm2)
        desired_tasks.add(task)

    # Load the summary table and complete tasks list, if they exist.
    if pathlib.Path(final_complete_tasks_filename).exists():
        final_complete_tasks = pickle.load(open(final_complete_tasks_filename, 'rb'))
    else:
        final_complete_tasks = set()
    if pathlib.Path(final_tables_filename).exists():
        combined_signature_table, combined_properties_table = pickle.load(open(final_tables_filename, 'rb'))
    else:
        combined_signature_table = None
        combined_properties_table = None

    # Join the results
    for dir in dirs:
        print("Looking at dir:", dir)
        p = pathlib.Path(dir)
        for child in p.iterdir():
            if not child.is_file():
                continue
            print("Looking at child:", child)
            if child.name == "complete_tasks.dat":
                done_tasks = pickle.load(open(child.as_posix(), 'rb'))
                final_complete_tasks = final_complete_tasks.union(set(done_tasks))
                pickle.dump(file=open(final_complete_tasks_filename, 'wb'), obj=final_complete_tasks)
            elif child.name.find("signature_table_worker") == 0:
                print("Loading signature and properties tables from file:", child.name)
                signature_table, properties_table = pickle.load(open(child.as_posix(), 'rb'))
                print(properties_table[0])
                if combined_properties_table is None:
                    combined_properties_table = properties_table
                    assert combined_signature_table is None, "What happened?"
                    combined_signature_table = signature_table
                else:
                    # I messed up... I cannot disambiguate a zero meaning "not yet computed" from a zero meaning the
                    # first entry in the lookup table.  Fortunately, I think all zeros mean "GAME_OVER" since the first
                    # arm will always be (-1,-1,-1) for any task.  I can pass once more over zeros at the end, I guess
                    # Form a mapping from one property table to another
                    print("Combining properties_tables.")
                    for property in properties_table:
                        if property not in combined_properties_table:
                            combined_properties_table.append(property)
                    current_property_index_to_combined = [combined_properties_table.index(property) for property in properties_table]
                    print("Remap signature_table to the combined property table...")
                    remapped_signature_table = numpy.array(current_property_index_to_combined)[signature_table]
                    print("Find non-zero entries...")
                    index = numpy.nonzero(remapped_signature_table)
                    print("Assign non-zero entries...")
                    combined_signature_table[index] = remapped_signature_table[index]  # Assign all non-zero values
                    # Should I feel confident that the mappings don't overlap?

            print("Saving final results to final.dat")
            final = (combined_signature_table, combined_properties_table)
            pickle.dump(file=open("final.dat", "wb"), obj=final)

    remaining = desired_tasks - final_complete_tasks
    print("All Done: {}, Remaining {}:\n{}".format(len(final_complete_tasks), len(remaining), remaining))

def main7():
    signature_table, properties_table = pickle.load(open("signature_tables.dat", "rb"))
    signatures = numpy.array([
            303081869,
            306723378,
            44801543,
            173941706,
        ])
    r = numpy.array(properties_table)[signature_table[signatures]]
    pprint.pprint(r)

def main8():
    signatures = [
            303081869,
            306723378,
            44801543,
            173941706,
        ]
    for signature in signatures:
        SIGNATURE_OFFSET = sum([3**i for i in range(18)])  # Add this to all signatures to make them >= 0.
        arms = blue_player.NextMoveClassifier.signature_index_to_arms(int(signature))
        print(signature, arms, SIGNATURE_OFFSET+blue_player.NextMoveClassifier.compute_signature(arms))
        signature, properties = blue_player.NextMoveClassifier.compute_signature_and_properties(arms)
        print(signature+SIGNATURE_OFFSET, properties)

def main9():
    game_so_far = ['g1', 'd5', 'e7', 'b5', 'e1', 'e3', 'i4', 'g4', 'b6', 'e4', 'f5', 'a5', 'c5', 'e8', 'c3', 'd3', 'f4', 'd2']
    # Moves by property:
    # SpaceProperies.WHITE_LOSE: ['c4', 'd6', 'f1', 'f3', 'f6']
    # ERROR: SpaceProperies.BLACK_SINGLE_CHECK: ['b1', 'b3', 'c6', 'd7', 'e6', 'g3'] # b3 is not a check... it is already blocked.
    # SpaceProperies.BLACK_WIN: ['d4']
    # SpaceProperies.BLACK_LOSE: ['c1', 'c2', 'd1', 'e2', 'e5', 'f2', 'f3']
    # ERROR: SpaceProperies.GAME_OVER: ['f7', 'h1']
    # SpaceProperies.WHITE_SINGLE_CHECK: ['d4', 'e5', 'f2']

    #game_so_far =  ['e8', 'a1', 'a2', 'a3', 'f1', 'a4', 'c1']
    # Moves by property:
    # SpaceProperies.BLACK_LOSE: ['a5']
    # game_so_far = ['d3']

    yavalath_engine.Render(board=yavalath_engine.HexBoard(), moves=game_so_far).render_image("debug.png")
    classifier = blue_player.NextMoveClassifier(game_so_far, verbose=True)
    classifier.compute_winning_and_losing_moves()


if __name__ == "__main__":
    #main9()
    # tasks = [
    #     ((1, 1, -1), (-1, 1, -1)),
    #     ((1, 1, -1), (0, -1, 0)),
    #     ((1, 1, -1), (1, -1, -1)),
    #     ((1, 1, 0), (-1, 0, 1)),
    #     ((1, 1, -1), (0, 0, 0)),
    #     ((1, 1, -1), (1, 0, -1))
    # ]
    #tasks=[((1, 1, -1), (0, 0, 0)), ((1, 1, -1), (1, 0, -1))]
    blue_player.NextMoveClassifier.compute_signature_table()
    #combine_results(dirs=[r"data",])
    #blue_player.NextMoveClassifier.compute_signature_table()
    #pprint.pprint(pickle.load(open("data/complete_tasks.dat", "rb")))