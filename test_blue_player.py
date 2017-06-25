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

def combine_results_simple(dirs = [r"data\laptop_files", r"data\desktop_files"]):
    """This one assumes all sub-results use the same properites dictionary, and that the sentinel value is -1"""
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
                    assert properties_table == combined_properties_table, "Everyone should be using the same table."

                    print("Computing index where values >= 0...")
                    index = numpy.where(signature_table >= 0)
                    print("Updating combined index...")
                    combined_signature_table[index] = signature_table[index]  # Assign all non-zero values
                    # Should I feel confident that the mappings don't overlap?

            print("Saving final results to final.dat")
            final = (combined_signature_table, combined_properties_table)
            pickle.dump(file=open("final.dat", "wb"), obj=final)

    remaining = desired_tasks - final_complete_tasks
    print("All Done: {}, Remaining {}:\n{}".format(len(final_complete_tasks), len(remaining), remaining))

    print("Counting sentinel values...")
    index = numpy.where(combined_signature_table == -1)
    print("Remaining sentinel values:", len(index[0]))


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

def regen_status():
    done_tasks = pickle.load(open("data/complete_tasks.dat", 'rb'))
    print("{} Local Done:\n{}".format(len(done_tasks), done_tasks))
    desktop_done = pickle.load(open(r"D:\yavalath\data/complete_tasks.dat", 'rb'))
    print("{} Desktop Done:\n{}".format(len(desktop_done), desktop_done))
    sig_table, prop_table = pickle.load(open("data/signature_table_worker_0.dat", "rb"))
    for index, p in enumerate(prop_table):
        print(index, list(i.value if i is not None else "None" for i in p))
    print("Max(sig_table):{}".format(sig_table.max()))
    h = numpy.histogram(sig_table, bins=[-1,] + list(range(30)))
    print(h)


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
    #blue_player.NextMoveClassifier.compute_signature_table()
    combine_results_simple(dirs=[r"data",])
    #blue_player.NextMoveClassifier.compute_signature_table()
    #pprint.pprint(pickle.load(open("data/complete_tasks.dat", "rb")))
    #regen_status()