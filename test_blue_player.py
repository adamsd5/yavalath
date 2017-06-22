import unittest
import yavalath_engine
from players import blue_player
import pprint

class TestDarrylPlayer(unittest.TestCase):
    def test_moves_to_vec(self):
        print(blue_player.moves_to_board_vector([]))
        print(blue_player.moves_to_board_vector(["e1"]))

    def test_vector_player(self):
        blue_player.vector_player([])


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

def main():
    import timeit
    game_so_far = "g1 d5 e7 b5 e1 e3 i4 g4 b6 e4 f5 a5 c5 e8 c3 d3 f4 d2 d4".split()
    print("Starting white spaces:", game_so_far[::2])
    print("Starting black spaces:", game_so_far[1::2])

    white_move_indices = {space_to_index[move] for move in game_so_far[::2]}
    black_move_indices = {space_to_index[move] for move in game_so_far[1::2]}

    t = timeit.timeit(lambda: get_board_hash(white_move_indices, black_move_indices))
    print("get_board_hash: 1M iterations took {}s".format(t))

    t = timeit.timeit(lambda: get_board_hash2(white_move_indices, black_move_indices))
    print("get_board_hash2: 1M iterations took {}s".format(t))

if __name__ == "__main__":
    main2()