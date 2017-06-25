#######################################################
# David Keith Maslen's Yavalath Players
#######################################################

import collections
import hashlib
import struct
import itertools
import time



#######################################################
# Board related library functions and data
#######################################################

STRING_POSITIONS = """
e5
e6 e7 e8 e9 d6 d7 d8 c7 c6 b6
d5 c5 b5 a5 c4 b4 a4 a3 b3 a2
d4 c3 b2 a1 d3 c2 b1 c1 d2 d1
e4 e3 e2 e1 f3 f2 f1 g1 g2 h1
f4 g3 h2 i1 g4 h3 i2 i3 h4 i4
f5 g5 h5 i5 f6 g6 h6 g7 f7 f8
""".split()

POS_2_STR = {i: x for i, x in enumerate(STRING_POSITIONS)}
STR_2_POS = {x: i for i, x in enumerate(STRING_POSITIONS)}

def str2pos(x):
    """ Converts a string position into a numerical position.

    Args:
        x:
            str. The string to convert. This consists of a letter
        from a to i indicating the vertical distance from the top of the board,
        and a number from 1 through 9 indicating the horizontal distance from the
        left most position on the same row. The letter may be upper or lower case.

    Returns:
        A number from 0 through 60 inclusive.
    """
    return STR_2_POS[x.lower()]


def pos2str(x):
    """ Converts a numerical position into a string position

    Args:
        x:
            int. The position number to convert. This is a number from 0 through 60
        inclusive.

    Returns:
        A string indication the postion on the board.
    """
    return POS_2_STR[x]


def splitbytwos(x):
    """Split a string into substrings of length two.

    Args:
        x:
            The string to split.

    Returns:
        A list of strings of length two
    """
    return [x[2*i:2*i+2] for i in range(len(x)//2)]


def sboard2pboard(sboard):
    """Converts a board defined as a list of strings alternating white and black positions
    to a board defined by a pair of lists of numerical positions.

    This preserves the black white ordering and so can be used to form representations
    of games.

    Args:
        sboard:
            list(str). A list of alternating white, black positions defined by strings

    Returns:
        A pair of lists of the white positions and the black positions.
    """
    white = list(str2pos(x) for i, x in enumerate(sboard) if i % 2 == 0)
    black = list(str2pos(x) for i, x in enumerate(sboard) if i % 2 == 1)
    return (white, black)


def pboard2sboard(pboard):
    """Converts a board defined by a pair of lists of numerical positions to a board
    defined by a list of strings alternating white and black positions.

    This preserves the black white ordering and so can be used to form representations
    of games.

    Args:
        pboard.
            A pair of lists indicating the white positions and the black positions.

    Returns:
        A list of strings describing a board by alternating white and black positions.
    """
    return ([pos2str(x) for pair in zip(pboard[0], pboard[1]) for x in pair] +
            [pos2str(x) for x in pboard[0][len(pboard[1]):]])


def positions2bits(positions):
    """Converts a list of numerical positions to an integer.

    Args:
        positions:
            list(int). A list of distinct integers in the range 0 through 60 inclusive.
    """
    return sum(1 << pos for pos in positions)


def str2positions(spattern):
    """Convert a pattern represented by a string into a pattern represented by a list of positions.

    Args:
        spattern:
            str. A pattern represented by a string. The positions are listed in string form
            with no spaces. E.g. a2a3a4

    Returns:
        A list of position numbers representing the same pattern.
    """
    return [str2pos(x) for x in splitbytwos(spattern)]


def positions2str(ppattern):
    """Convert a pattern represented by a list of position numbers into a pattern represented by a string.

    Args:
        ppattern:
            str. A pattern represented by a list of numbers.

    Returns:
        A string representing the same pattern. E.g. 'a1a2a3'
    """
    return ''.join(pos2str(x) for x in ppattern)


def str2bits(spattern):
    """Convert a pattern represented by a string into a pattern represented by an integer.

    Args:
        spattern:
            str. A pattern represented by a string. The positions are listed in string form
            with no spaces. E.g. a2a3a4

    Returns:
        An integer whose bits represent the pattern described by the string.
    """
    return positions2bits(str2positions(spattern))


def bits2str(bits):
    """Convert a pattern represented by a list of position numbers into a pattern represented by a string.

    Args:
        bits:
            int. An integer in the range 0 <= bits < 2 ** 61. The position of the nonzero bits in the
            integer represents the pattern.

    Returns:
        A string representing the same pattern. E.g. 'a1a2a3'
    """
    return positions2str(bits2positions(bits))


def bits2positions(bits):
    """Converts an integer in the range 0 <= bits < 2**61 to a list of integers
    in the range 0 through 60 inclusive indicating the positions of the nonzero bits.

    Args:
        bits:
            int. An integer in the range 0 <= bits < 2 ** 61

    Returns:
        A list of integers in the range 0 though 60 inclusive.
    """
    return list(i for i in range(61) if (2 ** i) & bits > 0)


def pboard2bboard(pboard):
    """Converts a board defined by a pair of lists of numerical positions to a bit board
    defined by a pair of integers, the bits of which indicate black and white positions.

    Args:
        pboard:
            A pair of lists of integers, the first indicating positions of white
            and the second indicating black positions.

    Returns:
        A pair of integers the bits of the first indicating white positions, and the
        bits of the second indicating black positions.
    """
    return (positions2bits(pboard[0]), positions2bits(pboard[1]))


def bboard2pboard(bboard):
    """Converts a bit board defined by a pair of integers into a board defined by a pair
    of lists of numerical positions.

    Args:
        bboard:
            A pair of integers the bits or which indicate white and black positions
        respectively.

    Returns:
        A pair of lists of integers, the first indicating positions of white
        and the second indicating black positions.
    """
    return (bits2positions(bboard[0]), bits2positions(bboard[1]))


def sboard2bboard(sboard):
    """Converts a board defined as a list of strings alternating white and black positions
    defined by a pair of integers, the bits of which indicate black and white positions.

    Args:
        sboard:
            list(str). A list of alternating white, black positions defined by strings.

    Returns:
        A pair of integers the bits of the first indicating white positions, and the
        bits of the second indicating black positions.
    """
    return pboard2bboard(sboard2pboard(sboard))


def bboard2sboard(bboard):
    """Converts a bit board defined by a pair of integers into a board defined by a
    list of strings alternating white and black positions.

    Args:
        bboard:
            A pair of integers the bits or which indicate white and black positions
        respectively.

    Returns:
        A list of strings describing a board by alternating white and black positions.
    """
    return pboard2sboard(bboard2pboard(bboard))


def board2string(board, marked=()):
    """Returns an ascii representation of a Yavalath board.  White is represented
    by O and black by X. Empty positions are represented by a period.

    Args:
        board:
            A representation of a Yavalath board either as a sboard, a pboard,
            or a bit board.

        marked:
            Either a string, an integer, a list of integers, or a list of strings, indicating
            positions to be highlighted. If an integer is passed the bits of the integer indicate
            the positions to be highlighted. If a string is passed then it will be grouped into
            substrings of length 2.

    Returns:
        A printable ascii representation of the board.
    """
#    template = """      1 2 3 4 5
# a      {24} {20} {18} {17} {14} 6
# b    {27} {23} {19} {16} {13} {10} 7
# c    {28} {26} {22} {15} {12} {9} {8} 8
# d  {30} {29} {25} {21} {11} {5} {6} {7} 9
# e  {34} {33} {32} {31} {0} {1} {2} {3} {4}
# f  {37} {36} {35} {41} {51} {55} {59} {60} 9
# g    {38} {39} {42} {45} {52} {56} {58} 8
# h    {40} {43} {46} {49} {53} {57} 7
# i      {44} {47} {48} {50} {54} 6
#        1 2 3 4 5"""
#    """
    template = """        1 2 3 4 5
      a {24} {20} {18} {17} {14} 6
     b {27} {23} {19} {16} {13} {10} 7
    c {28} {26} {22} {15} {12} {9} {8} 8
   d {30} {29} {25} {21} {11} {5} {6} {7} 9
  e {34} {33} {32} {31} {0} {1} {2} {3} {4} e
   f {37} {36} {35} {41} {51} {55} {59} {60} 9
    g {38} {39} {42} {45} {52} {56} {58} 8
     h {40} {43} {46} {49} {53} {57} 7
      i {44} {47} {48} {50} {54} 6
        1 2 3 4 5"""

    # Handle different input formats for boards
    board = sboard2pboard(board) if len(board) == 0 or isinstance(board[0], str) else board
    board = bboard2pboard(board) if isinstance(board[0], int) else board
    # Handle different input formats for marked positions
    marked = str2positions(marked) if isinstance(marked, str) else marked
    marked = bits2positions(marked) if isinstance(marked, int) else marked
    marked = [str2pos(x) for x in marked] if len(marked) > 0 and isinstance(marked[0], str) else marked
    # Compute the list of symbols to pass into the template and create the string
    white = set(board[0])
    black = set(board[1])
    symbols = ['o' if x in white else 'x' if x in black else '-' for x in range(61)]
    symbols = [x.upper() if x != '-' and i in marked else x for i, x in enumerate(symbols)]
    return template.format(*symbols)


def print_board(board, marked=()):
    """Prints an ascii representation of a Yavalath board. White is represented
    by O and black by X. Empty positions are represented by a period.

    Args:
        board:
            A representation of a Yavalath board either as a sboard, a pboard,
            or a bit board.

        marked:
            Either a string, an integer, a list of integers, or a list of strings, indicating
            positions to be highlighted. If an integer is passed the bits of the integer indicate
            the positions to be highlighted. If a string is passed then it will be grouped into
            substrings of length 2.

    Returns:
        None
    """
    print(board2string(board, marked))


def rotate(bits, i):
    """Rotate a bit board or bit pattern by i sixty degree counter clockwise rotations.

    Args:
        bits:
            An integer whose bits represent a pattern on a Yavalath board, or a tuple of integers.

        i:
            An integer indicating the number of sixty degree counter clockwise rotations
            to apply. Negative numbers indicate clockwise rotations.

    Returns:
        The rotated bit board or bit pattern
    """
    i = i % 6
    if i == 0:
        return bits
    if isinstance(bits, int):
        shift = 10 * i
        # 2**61 - 2 == (1 << 61) - 2 == 0x1ffffffffffffffe
        return ((bits & (1 << 61 - shift) - 2) << shift) + (bits >> 60 - shift &0x1ffffffffffffffe) + (bits & 1)
    return tuple(rotate(x, i) for x in bits)


def flip(bits, i):
    """Flip a bit board or bit pattern about an axis that is i * 30 degrees counter clockwise
    from the horizontal axis.

    Args:
        bits:
            An integer whose bits represent a pattern on a Yavalath board, or a pair of integers
            whose bits represent the white and black positions.

        i:
            An integer indicating the number of counter clockwise 30 degree rotations
            between the horizontal axis and the axis of the flip.

    Returns:
        The flipped bit board.
    """
    # Do a horizontal flip by converting to a sboard or spattern then flipping the letters.
    sboard = ([pos2str(x) for x in bits2positions(bits)] if isinstance(bits, int) else
                bboard2sboard(bits))
    flipped_sboard = [chr(202 - ord(x[0])) + x[1] for x in sboard]
    flipped = (positions2bits([str2pos(x) for x in flipped_sboard]) if isinstance(bits, int) else
              sboard2bboard(flipped_sboard))
    return rotate(flipped, i % 6)

def dihedral(bits, f, i):
    """Apply an element of the dihedral group to a bit board or bit pattern.

    Args:
        bits:
            An integer whose bits represent a pattern on a Yavalath board, or a tuple of integers

        f:
            An integer indicating the number of flips about the horizontal axis to perform.

        i:
            An integer indicating the number of sixty degree counter clockwise rotations
            to apply. Negative numbers indicate clockwise rotations.

    Returns:
        The rotated or flipped bit board or bit pattern. The flip about the horizontal axis, if
        any, is done before the rotation.

    """
    if isinstance(bits, int):
        return flip(bits, i) if f % 2 else rotate(bits, i)
    return tuple(dihedral(x, f, i) for x in bits)


def to_fundamental_domain(bits):
    """Maps an bit pattern or bit board to the smallest that is equivalent under the action of the dihedral group.

    Args:
        bits:
            An integer whose bits represent a pattern on a Yavalath board, or a pair of integers
            whose bits represent the white and black positions.

    Returns:
        A triple consisting of the transformed bit pattern or bit board, the number of flips about the horizontal and the number of 60 degree rotations required to map the input puts onto the output. The flips are applied before the rotations.
    """
    return min((dihedral(bits, f, i), f, i) for f in range(2) for i in range(6))


def orbit(patterns):
    """Calculate the set of all patterns or bit boards generated by applying all elements of the
    dihedral group to all objects in patterns.

    Args:
    patterns:
        A list of bit boards or bit patterns.

    Returns:
        A list containing all bit patterns or bit boards generated by applying all elements of
        the dihedral group to the patterns or bit boards in boards.
    """
    return sorted(set(dihedral(x, f, i) for x in patterns for f in range(2) for i in range(6)))


def patterns_by_position(patterns):
    return {pos: [pattern for pattern in patterns if pattern & (1 << pos) == (1 << pos)] for pos in range(61)}

#######################################################
# Patterns
#######################################################

# Terminology:
# a three = a line of three
# a four  = a line of four
# a split four = a line of four with something else in the middle
# a wide four = a line of four with two other things in the middle
# a check = a line of four with an empty position in the middle in one position
# an open = a line of four with two empty positions in the middle

# Common bit patterns and board, be careful not to modify these
EMPTY = 0
FULL = (1 << 61) - 1
EMPTY_BOARD = (0, 0)


# Some string constants used to generate collections of bit patterns
SOME_THREES = """a1a2a3 a2a3a4 a3a4a5 b1b2b3 b2b3b4 b3b4b5 b4b5b6
c1c2c3 c2c3c4 c3c4c5 c4c5c6 c5c6c7 d1d2d3 d2d3d4 d3d4d5 d4d5d6 d5d6d7 d6d7d8
e1e2e3 e2e3e4 e3e4e5 e4e5e6 e5e6e7 e6e7e8 e7e8e9""".split()

SOME_FOURS = """a1a2a3a4 a2a3a4a5 b1b2b3b4 b2b3b4b5 b3b4b5b6
c1c2c3c4 c2c3c4c5 c3c4c5c6 c4c5c6c7 d1d2d3d4 d2d3d4d5 d3d4d5d6 d4d5d6d7 d5d6d7d8
e1e2e3e4 e2e3e4e5 e3e4e5e6 e4e5e6e7 e5e6e7e8 e6e7e8e9""".split()

SOME_FIVES = """a1a2a3a4a5 b1b2b3b4b5 b2b3b4b5b6 c1c2c3c4c5 c2c3c4c5c6 c3c4c5c6c7
d1d2d3d4d5 d2d3d4d5d6 d3d4d5d6d7 d4d5d6d7d8 e1e2e3e4e5 e2e3e4e5e6 e3e4e5e6e7 e4e5e6e7e8 e5e6e7e8e9""".split()

SOME_SIXES = """b1b2b3b4b5b6 c1c2c3c4c5c6 c2c3c4c5c6c7 d1d2d3d4d5d6 d2d3d4d5d6d7 d3d4d5d6d7d8
e1e2e3e4e5e6 e2e3e4e5e6e7 e3e4e5e6e7e8 e4e5e6e7e8e9""".split()

SOME_EDGE_FOURS = """a1a2a3a4 b1b2b3b4 c1c2c3c4 d1d2d3d4 e1e2e3e4""".split()

SOME_SPLIT_FOURS = [x[:2] + x[4:8] for x in SOME_FOURS]
SOME_SPLIT_FOUR_GAPS = [x[2:4] for x in SOME_FOURS]
SOME_WIDE_FOURS = [x[:2] + x[6:8] for x in SOME_FOURS]
SOME_WIDE_FOUR_GAPS = [x[2:6] for x in SOME_FOURS]
SOME_WIDE_FIVES = [x[:2] + x[8:10] for x in SOME_FIVES]
SOME_WIDE_SIXES = [x[:2] + x[10:12] for x in SOME_FIVES]
SOME_SKEW_TRAPS = [x[2:6] for x in SOME_FIVES]
SOME_SKEW_TRAP_GAPS = [x[6:8] for x in SOME_FIVES]
SOME_FIVE_TRAPS = [x[2:4] + x[6:8] for x in SOME_FIVES]
SOME_FIVE_TRAP_GAPS = [x[4:6] for x in SOME_FIVES]


SOME_SIX_TRAPS = [x[4:8] for x in SOME_SIXES]
SOME_SIX_TRAP_GAPS = [x[2:4] + x[8:10] for x in SOME_SIXES]




# Collections of bit patterns
ALL_THREES = orbit(set(to_fundamental_domain(str2bits(x))[0] for x in SOME_THREES))
ALL_FOURS = orbit(set(to_fundamental_domain(str2bits(x))[0] for x in SOME_FOURS))
ALL_SPLIT_FOURS = orbit(set(to_fundamental_domain(str2bits(x))[0] for x in SOME_SPLIT_FOURS))
ALL_WIDE_FOURS = orbit(set(to_fundamental_domain(str2bits(x))[0] for x in SOME_WIDE_FOURS))

# Collections of bit-board-like patterns. However these do not represent black and white positions,
# but more complicated things such as presence of a white and absence of a black.
ALL_CHECKS = [ (split, gap, bits2positions(gap)[0]) for split, gap in
              orbit(set(to_fundamental_domain((str2bits(x), str2bits(y)))[0]
                      for x, y in zip(SOME_SPLIT_FOURS, SOME_SPLIT_FOUR_GAPS)))]

ALL_OPENS = orbit(set(to_fundamental_domain((str2bits(x), str2bits(y)))[0]
                      for x, y in zip(SOME_WIDE_FOURS, SOME_WIDE_FOUR_GAPS)))

ALL_SKEW_TRAPS = orbit(set(to_fundamental_domain((str2bits(x), str2bits(y), str2bits(z)))[0]
                           for x, y, z in zip(SOME_WIDE_FIVES, SOME_SKEW_TRAPS, SOME_SKEW_TRAP_GAPS)))

ALL_TRAPS = ALL_SKEW_TRAPS

# Patterns organized by position on board
THREES_BY_POSITION = patterns_by_position(ALL_THREES)
FOURS_BY_POSITION = patterns_by_position(ALL_FOURS)
CHECKS_BY_POSITION = {pos: [(split, gap, gpos) for split, gap, gpos in ALL_CHECKS if (split|gap) & (1 << pos) == (1 << pos)] for pos in range(61)}
OPENS_BY_POSITION = {pos: [(wide, gap) for wide, gap in ALL_OPENS if (wide|gap) & (1 << pos) == (1 << pos)] for pos in range(61)}
TRAPS_BY_POSITION = {pos: [(wide, trap, gap) for wide, trap, gap in ALL_TRAPS if (wide|trap|gap) & (1 << pos) == (1 << pos)] for pos in range(61)}


SPLIT_FOURS_BY_POSITION = patterns_by_position(ALL_SPLIT_FOURS)
WIDE_FOURS_BY_POSITION = patterns_by_position(ALL_WIDE_FOURS)
CHECKS_BY_SPLIT_POSITION = {pos: [(split, gap, gpos) for split, gap, gpos in ALL_CHECKS if split & (1 << pos) == (1 << pos)] for pos in range(61)}
CHECKS_BY_GAP_POSITION = {pos: [(split, gap, gpos) for split, gap, gpos in ALL_CHECKS if gap & (1 << pos) == (1 << pos)] for pos in range(61)}

OPENS_BY_WIDE_POSITION = {pos: [(wide, gap) for wide, gap in ALL_OPENS if wide & (1 << pos) == (1 << pos)] for pos in range(61)}
OPENS_BY_GAP_POSITION = {pos: [(wide, gap) for wide, gap in ALL_OPENS if gap & (1 << pos) == (1 << pos)] for pos in range(61)}



#######################################################
# Game utility functions. These are not intended to be optimized for speed.
#######################################################

WHITE_WIN = 0
BLACK_WIN = 1
DRAW = 2
GAME_NOT_CONCLUDED = 3
INVALID_BOARD = 4

WHITE_OR_BLACK = ('white', 'black')

def choice(choices, i, seed=b'q32do7ytf5o8i2fyh5tiuldfyilh'):
    h = hashlib.new('sha1')
    h.update(seed + bytes(repr(i), 'utf-8'))
    i = struct.unpack('q', h.digest()[:8])[0] % len(choices)
    return choices[i]

def board_turn(board):
    return bin(board[0] | board[1]).count('1')


def player_to_move(board):
    return board_turn(board) % 2


def list_game_boards(maxlen, exact=True):
    """Returns a list of triples (board, game, result).
    """
    # There is exactly one board of length 0
    if maxlen == 0:
        return [((0, 0), (), GAME_NOT_CONCLUDED)]
    # Get the list of boards of length maxlen - 1 or less
    shorter = list_game_boards(maxlen - 1)
    # Generate a list of boards of length n
    longer = []
    player = maxlen % 2
    for board, game, result in shorter:
        if len(game) < maxlen - 1 or result != GAME_NOT_CONCLUDED:
            continue
        occupied = board[0] | board[1]
        for move in range(61):
            if (1 << move) & occupied == 0:
                newboard = move_board(board, move, player)
                strmove = pos2str(move)
                longer.append((newboard, game + (strmove,), winlosedraw(newboard),
                              to_fundamental_domain(newboard)[0]))
    # Remove duplicate boards and boards related by symmetry, and sort.
    longer.sort()
    seen = set()
    unique = []
    for board, game, result, representative in longer:
        if representative in seen:
            continue
        unique.append((board, game, result))
        seen.add(representative)
    return unique if exact else shorter + unique


def unfinished_games(depth):
    return [game[1] for game in list_game_boards(depth) if game[2] == GAME_NOT_CONCLUDED]


def move_board(board, move, player):
    """Calculate the board obtained by making a move by the given player.

    Args:
        board:
            A bitboard.

        move:
            An integer position, or None for no move.

        player:
            A player number
    """
    bitmove = 1 << move if move is not None else 0
#    occupied = board[0] | board[1]
#    if board[0] & bitmove * (1 - player) or board[1] & bitmove * player:
#        raise ValueError('{} is an invalid move from parent board {}'.format(move, board))
    return (board[0] | bitmove * (1 - player), board[1] | bitmove * player)


def related_by_move(parent, move, child):
    """Test whether the child is related to the parent by move move."""
    if move is None:
        return parent == child
    if player_to_move(parent) == 0:
        return child[1] == parent[1] and (parent[0] | (1 << move)) == child[0]
    return child[0] == parent[0] and (parent[1] | (1 << move)) == child[1]


def result_string(result):
    """Prints out a human readable string describing the state of play.

    Args:
        result:
            An integer in the range 0 through 4 inclusive. 0 for a white win,
            1 for a black win, 2 for a draw, 3 for a game not yet concluded and 4
            for an impossible board.

    Returns:
        A string decribing the result.
    """
    result_string = ['win for white', 'win for black', 'draw', 'not concluded', 'invalid game']
    return result_string[result]


def valid_move(board, move):
    """Test whether a move is valid by checking if is matched an occupied position

    Args:
        board:
            A bitboard

        move:
            An int or str representing the position of the move. The int is a position number.

    Returns:
        True or False depending on whether the move is valid.
    """
    move = str2pos(move) if isinstance(move, str) else move
    return (board[0] | board[1]) & (1 << move) == 0


def winlosedraw(bboard):
    """Calculate whether a board represents a win a lose or a draw or the game as not yet finished.

    Args:
        bboard:
            A bit board.

    Returns:
        An integer indicating whether the game is a win lose or a draw. 0 is returned
        for a white win, 1 for a black win, 2 for a draw, 3 for a game that has not yet
        finished, and 4 for an impossible game where a board representing both a white
        and black win is detected.
    """
    white, black = bboard
    white_threes = sum(1 for x in ALL_THREES if x & white == x)
    white_fours = sum(1 for x in ALL_FOURS if x & white == x)
    black_threes = sum(1 for x in ALL_THREES if x & black == x)
    black_fours = sum(1 for x in ALL_FOURS if x & black == x)
    white_win = white_fours > 0 or (black_threes > 0 and not black_fours > 0)
    black_win = black_fours > 0 or (white_threes > 0 and not white_fours > 0)
    end_of_game = white | black == FULL
    win = 2 * (1 - white_win) + (1 - black_win) # 0 both, 1 white, 2 black, 3 neither
    return 4 if win == 0 else win - 1 if end_of_game else win - 1 + (win == 3)




#######################################################
# Yavalath player classes and factories
#######################################################


#######################################################
# A basic random player
#######################################################

class YavaRandom1(object):
    """A Yavalath player object implementing random play
    """

    def __init__(self, seed=b'123456', verbose=False):
        # Algorithm parameters
        self.seed = seed
        self.verbose = verbose
        # Properties that vary from game to game but are fixed through the game
        self.player = None

        # A record of the game and other player state
        self.game_moves = None
        self.game_stats = None
        self.current = None
        self.allowed = None
        self.choice_count = None
        self.principal_variation = []


    def start_game(self, player):
#        print('Starting new game')
        self.player = player
        self.game_moves = []
        self.game_stats = []
        self.current = EMPTY_BOARD
        self.allowed = set(range(61))
        self.principal_variation = []


    def update(self, move):
        # Ignore a repeated message. This helps the player handle different protocols more easily.
        if self.game_moves and move == self.game_moves[-1]:
            return
        # Do the update
        player = len(self.game_moves) % 2 # Player 0 is white
        self.game_moves.append(move)
        posmove = str2pos(move)
        bitmove = str2bits(move)
        self.current = move_board(self.current, posmove, player)
#        print('Removing move from allowed set', posmove)
        self.allowed = self.allowed - {posmove}


    def your_move(self):
        # Try random moves, but don't lose the game on this move unless forced to,
        # and win the game if obvious.
        available = sorted(self.allowed)
        # Calculate wins and losses
        values = [winlosedraw(move_board(self.current, m, self.player)) for m in available]
        wins = [m for m, v in zip(available, values) if v == self.player]
        loses = [m for m, v in zip(available, values) if v == 1 - self.player]
        draws = [m for m, v in zip(available, values) if v == DRAW]
        choices = [m for m, v in zip(available, values) if v == GAME_NOT_CONCLUDED]
        best_moves = wins or draws or choices or loses
        move = None if not best_moves else choice(best_moves, 0, self.seed + bytes(repr(self.game_moves), 'utf-8'))
        str_move = pos2str(move) if move is not None else None
        self.principal_variation = [str_move]
        if str_move is not None:
            self.update(str_move)
        return str_move

    def get_stats():
        return self.game_stats




#######################################################
# A basic negamax based player
#######################################################

# The YavaNaga1 payer uses records. For now we will use lists for these.
# The contents of the records are as follows
#
# item  contents
# 0      A bitboard corresponding to the record
# 1      Features computed from the bit-board
# 2      The heuristic value of the board for the player whose turn it would be
# 3      The estimated value of the board from searches
# 4      Accuracy flag. 1 for exact, 2 for lower bound, 3 for upper bound
# 5      Best move. The move corresponding to the best value.
# 6      The minimum height below the board that has been searched
# 7      The maximum height below the board that has been searched
# 8      The minimum depth at which the best value can be forced
# 9      The maximum depth at which the best value can be forced

#
# A summarizer takes a bit board, a parent record, a player number, move position number
# and returns a features record. These should be such that the bit board is that which
# would be obtained from the parent if the player with the given player number moved at
# the given move position. The player number and move position number are therefore only
# present to avoid recalculation of those values. If an empty board is passed to a summarizer
# then the parent, player number, and move number are ignored, and the features for an
# empty board are returned.
#
# An evaluator takes a bit board, a features record, and returns a score for the
# bitboard assuming the specified player is to move next. Note that the convention for values is
# that the value for a board is the value the player whose turn it is would assign the board. So
# the normal case would be to pass in the player whose turn it is. This could of course be determined
# from the board.
#

EXACT_VALUE = 1
LOWER_BOUND = 2
UPPER_BOUND = 3

YavaFeatures1 = collections.namedtuple(
    'YavaFeatures1',
    ['white_threes', 'white_fours', 'white_check_1', 'white_check_2', 'white_opens', 'white_traps',
    'black_threes', 'black_fours', 'black_check_1', 'black_check_2', 'black_opens', 'black_traps',
    'unused_12', 'unused_13', 'unused_14', 'unused_15',
    ])

BOARD_FORMAT = 'qq'
FEATURE_FORMAT = 'bbbbbbbbbbbbbbbb'
VALUES_FORMAT = 'iibbbbbbbb' # The heuristic is included here for alignment reasons
RECORD_FORMAT = BOARD_FORMAT + FEATURE_FORMAT + VALUES_FORMAT

FEATURES_START_INDEX = 2
FEATURES_END_INDEX = FEATURES_START_INDEX + len(YavaFeatures1._fields)

BoardStruct = struct.Struct(BOARD_FORMAT)
FeatureStruct = struct.Struct(FEATURE_FORMAT)
ValuesStruct = struct.Struct(VALUES_FORMAT)
RecordStruct = struct.Struct(RECORD_FORMAT)
VALUES_START_OFFSET = BoardStruct.size + FeatureStruct.size

def pack_record(record, buffer):
    RecordStruct.pack_into(buffer, 0, *(record[0] + record[1] + record[2:-1])) # Slow


def unpack_record(packed):
    unpacked = RecordStruct.unpack(packed)
    board = unpacked[:FEATURES_START_INDEX]
    features = YavaFeatures1._make(unpacked[FEATURES_START_INDEX: FEATURES_END_INDEX])
    return (board, features) + unpacked[FEATURES_END_INDEX:] + (packed,) # Slow. Can be improved


def pack_values(buffer, heuristic, value, accuracy, best_move, minhgt, maxhgt, bestminhgt, bestmaxhgt):
    # The heuristic is included here for alignment reasons
    ValuesStruct.pack_into(buffer, VALUES_START_OFFSET, heuristic,
                          value, accuracy, best_move, minhgt, maxhgt, bestminhgt, bestmaxhgt,
                          0, 0)

def child_record(parent, move, summarizer, evaluator, table, counters=[0] * 4):
    bitmove = 1 << move if move is not None else 0
    occupied = parent[0][0] | parent[0][1]
    player = bin(occupied).count('1') % 2
    board = move_board(parent[0], move, player)
    # Extra check
    if bitmove & occupied:
        raise ValueError('{} is an invalid move for board {}'.format(move, parent[0]))

    key = BoardStruct.pack(*board)
    buffer = None if table is None else table.get(key)
    if buffer is None:
        counters[3] += 1
        features = summarizer(board, parent, player, move)
        value = evaluator(board, features)
        buffer = bytearray(RecordStruct.size)
        record = (board, features, value, value, EXACT_VALUE, -1, 0, 0, 0, 0, 0, 0, buffer)
        pack_record(record, buffer)
        if table is not None:
            table[key] = buffer
    else:
        record = unpack_record(buffer)
    return record


def clean_table(table, board, player):
    pattern = board[player]
    for key in list(table.keys()):
        tboard = struct.unpack('qq', key)
        if tboard[player] & pattern != pattern:
            del table[key]


def find_principal_variation(board, table):
    variation = []
    while True:
        key = BoardStruct.pack(*board)
        buffer = table.get(key)
        if buffer is None:
            break
        record = unpack_record(buffer)
        move = record[5]
        if move == -1:
            break
        variation.append(move)
        board = move_board(board, move, player_to_move(board))

    return variation

# Collection and display of algorithm statistics
YavaStats = collections.namedtuple('YavaStats', ['time', 'move', 'value', 'pv', 'mindepth', 'maxdepth',
                                                'searched', 'leaves', 'cuts', 'forced', 'forcing',
                                                'evals', 'table_size'])

# The Negamax player

class YavaNega1(object):
    """A Yavalath player object implementing a negamax algorithm.
    """

    def __init__(self, summarizer, evaluator, algorithm, mindepth, maxdepth=None, seed=b'123456', deepen=False,
                force=False, verbose=False, clean=True):
        # Algorithm parameters
        self.mindepth = mindepth
        self.maxdepth = mindepth if maxdepth is None else maxdepth
        self.summarizer = summarizer  # Calculates stats from a board, it's parent, and the newest position
        self.evaluator = evaluator    # Calculates a score from a board and it's stats
        self.algorithm = algorithm    # The algorithm to use
        self.seed = seed
        self.deepen = deepen
        self.force = force
        self.verbose = verbose
        self.clean = clean

        # Properties that vary from game to game but are fixed through the game
        self.player = None

        # A record of the game
        self.game_moves = None
        self.game_stats = None
        self.current = None
        self.previous = None
        self.table = None
        self.depth = None
        self.principal_variation = []


    def start_game(self, player):
        self.player = player
        self.game_moves = []
        self.game_stats = []
        self.previous = None
        self.table = {}
        self.depth = 0
        # Initialize an empty board with some stats
        self.current = child_record((EMPTY_BOARD, None, 0, 0, EXACT_VALUE, 0, 0, 0, 0, 0, 0, 0, None),
                                    None, self.summarizer, self.evaluator, self.table)
        self.allowed = set(range(61))
        self.principal_variation = []


    def update(self, move):
        # Ignore a repeated message. This helps the player handle different protocols more easily.
        if self.game_moves and move == self.game_moves[-1]:
            return

        # Otherwise validate the message against what is expected
        if move in self.game_moves:
            raise ValueError('Move {} has already occurred in the game')

        # Do the update
        player = len(self.game_moves) % 2 # Player 0 is white
        self.game_moves.append(move)
        self.previous = self.current
        posmove = str2pos(move)
        self.current = child_record(self.previous, posmove, self.summarizer, self.evaluator,
                                    self.table)
        self.allowed = self.allowed - {posmove}
        # Clean the table to remove entries that do not correspond to this move
        if self.clean:
            clean_table(self.table, self.current[0], player)


    def your_move(self):
        # Start a clock running
        start = time.time()

        # If we have the first move then restrict the allowed moves to moves 0 through 8 inclusive
        # This should cut the initial computation time.
        allowed = self.allowed if self.game_moves else {0, 1, 2, 3, 4, 5, 6, 7, 8}

        # Find the last move and convert it into different forms, and also find the current board
        last_move_str = self.game_moves[-1] if len(self.game_moves) > 0 else None
        last_move_pos = str2pos(last_move_str) if last_move_str is not None else None

        #Now we want to improve our valuation of self.current and get a recommended move
        # Initial iterative deepening
        counters = [0, 0, 0, 0, 0, 0] # searched, leaves, cuts
        if self.deepen and self.depth < self.mindepth:
#            print('Initial deepening to depth', self.depth)
            pv = self.algorithm(self.current, self.player, allowed, self.summarizer, self.evaluator, self.table, counters, self.depth, self.maxdepth, self.force, self.seed)
            self.current = unpack_record(self.current[-1]) # Update with data from the table
            self.depth = self.depth + 1
        # Call the search algorithm to find the principal variation
        pv = self.algorithm(self.current, self.player, allowed, self.summarizer, self.evaluator, self.table, counters, self.mindepth, self.maxdepth, self.force, self.seed)
        self.current = unpack_record(self.current[-1]) # Update with data from the table

        # Convert to string moves
        principal_variation = [pos2str(move) for move in pv]
        str_move = principal_variation[0] if principal_variation else None

        # Record the table size before updating the table
        table_size = len(self.table)
        searched = table_size

        # Perform a final validation
        if str_move in self.game_moves:
            raise ValueError('Program logic error: Invalid move {} that has already occurred was generated. Game so far: {}'.format(str_move, ' '.join(self.game_so_far)))

        # Update the state of the game to reflect the new move
        self.principal_variation = principal_variation
        if str_move is not None:
            self.update(str_move)

        # Compute the time taken and other statistics
        end = time.time()
        time_taken_for_move = end - start

        # Update statistics
        if str_move is not None:
            stats = YavaStats(time=time_taken_for_move,
                                  move=str_move,
                                  value=self.previous[3],
                                  pv=principal_variation,
                                  mindepth=self.previous[6],
                                  maxdepth=self.previous[7],
                                  searched=counters[0],
                                  leaves=counters[1],
                                  cuts=counters[2],
                                  evals=counters[3],
                                  forced=counters[4],
                                  forcing=counters[5],
                                  table_size=table_size)
            self.game_stats.append(stats)

        if self.verbose:
            if str_move is not None:
                print('Principal variation:', ' '.join(stats.pv))
                print('Statistics: {:.2f} {} {} {} {} {} {} {} {} {} {}'.format(stats.time, stats.value, stats.mindepth, stats.maxdepth, stats.searched, stats.leaves, stats.cuts, stats.forced, stats.forcing, stats.evals, stats.table_size), flush=True)
#                print(flush=True)
            else:
                print('No available moves.')

        # Return the result
        return str_move


    def get_stats(self):
        return self.game_stats


##############################################
# Valuation constants
##############################################

MAX_VALUE = 1000000000
MIN_VALUE = -MAX_VALUE
WIN_THRESHOLD = MAX_VALUE - 10000000
WIN_VALUE = WIN_THRESHOLD + 5000000
LOSE_THRESHOLD = MIN_VALUE + 10000000
LOSE_VALUE = LOSE_THRESHOLD - 5000000


##############################################
# Negamax
##############################################

def negamax(record, player, available, summarizer, evaluator, table, counters,
            mindepth, maxdepth, force, seed):
    board, features, heuristic, value, accuracy, best, minhgt, maxhgt, bestminhgt, bestmaxhgt, *extra  = record
    counters[0] += 1
    # Use a list for available, and enforce some determinism on the order
    available = sorted(available)
    # Check whether we have already lost or drawn
    current = record[0]
    if winlosedraw(current) != GAME_NOT_CONCLUDED:
        counters[1] += 1
        return []

    # Depth 0 search means no search. So choose a deterministically random move
    if mindepth < 1:
        counters[1] += 1
        return [choice(available, 0, seed + bytes(repr(board), 'utf-8'))]

    # Find the currently estimated values of the child nodes to the next player
    child_records = [child_record(record, move, summarizer, evaluator, table, counters) for move in available]

    # Order the child nodes with the highest values first
    children = sorted(((-r[3], -move, move, r) for r, move in zip(child_records, available)), reverse=True)
#    print('children:', ' '.join(pos2str(move) for _, move, _ in children))

    # Look for the move with best value
    white, black = current
    occupied = white | black
    best_value = MIN_VALUE
    best_move = None
    best_child = None
    alpha = MIN_VALUE
    beta = MAX_VALUE # Maybe use WIN_THRESHOLD instead?
    for _, _, move, child in children:
        # Debugging: Check the child record has the correct relation to the parent
        if not related_by_move(current, move, child[0]):
            raise ValueError('Parent {} is not related to child {} by move {}'.format(current, child[0], move))
        moves_available_to_child = [x for x in range(61) if (1 << x) & occupied == 0 and x != move]
        move_value = -negamax_value(child, 1 - player, moves_available_to_child,
                                    summarizer, evaluator, table, counters, mindepth, maxdepth, -beta, -alpha, force, 1, 0)
        # It is important to only accept the first best move because the later ones may
        # be only lower bounds due to pruning. This is because we use a weak beta cutoff,
        # which enables us to handle coarser evaluation functions.
        if move_value > best_value:
            best_value = move_value
            best_move = move
            best_child = child

        alpha = max(alpha, move_value) # This is the best value we have seen so far

    # Sanity check
    if abs(heuristic) >= WIN_THRESHOLD and abs(best_value) < WIN_THRESHOLD:
        raise ValueError('Heuristic is definite win or loss but value is not')

    # Fetch the new child information from the table
    best_child = unpack_record(best_child[-1])
#    if best_child[6] > 20:
#        raise ValueError('best child depth searched was {}'.format(best_child[6])

    # Record the results in the transposition table
    buffer = record[-1]
    pack_values(buffer,
                heuristic, # The heuristic is included here for alignment reasons
                best_value, # best_value
                EXACT_VALUE, # accuracy
                best_move, # The move corresponding to the best value
                mindepth,  # minhgt
                best_child[7] + 1, # maxhgt
                0, # rextension
                0) # bestmaxhgt

    # Calculate the principal variation
    child_variation = find_principal_variation(best_child[0], table)
    principal_variation = [best_move] + child_variation

    return principal_variation


def negamax_value(record, player, available, summarizer, evaluator, table, counters,
                  mindepth, maxdepth, alpha, beta, force, depth, extension):
    board, features, heuristic, value, accuracy, best, minhgt, maxhgt, rextension, bestmaxhgt, *extra  = record
    counters[0] += 1
    white, black = board

    # First check if this node has already been searched to the required height
    original_alpha = alpha
    required_height = mindepth + extension - depth
    if minhgt >= required_height and extension >= rextension:
        if accuracy == EXACT_VALUE:
            return value
        elif accuracy == LOWER_BOUND:
            alpha = max(alpha, value)
        else:
            beta = min(beta, value)
        if alpha > beta:
            return value

    # If we have searched to the required depth or the values are terminal or the game is finished
    # the return the value of the node as given.
    if depth == mindepth + extension or abs(heuristic) > WIN_THRESHOLD or white | black == FULL:
        counters[1] += 1
        return value

    if not available:
        raise ValueError('No available moves for non-empty board')

    moves_to_check = available
    if force:
        force_move = (features.black_check_1 if player == 0 and features.white_check_1 < 0 else
                      features.white_check_1 if player == 1 and features.black_check_1 < 0 else -1)
        i_am_forced = (force_move >= 0)
        if i_am_forced:
            moves_to_check = [force_move]

    # Find the currently estimated values of the child nodes to the next player
    # board, features, value, minhgt, maxhgt
    child_records = [child_record(record, move, summarizer, evaluator, table, counters) for move in moves_to_check]
    # Order the child nodes with the highest values first
    children = sorted(((-r[3], -move, move, r) for r, move in zip(child_records, moves_to_check)), reverse=True)
#    if force and i_am_forced:
#        children = [child for child in children if child[1] == force_move] # If we do not take this move we will lose immediately

    # Look for the child with the best negamax value
    best_value = MIN_VALUE
    best_move = None
    best_child = None
    new_maxhgt = required_height

    for _, _, move, child in children:
        extend = 0
        if force:
            if required_height == 1:
                i_am_forcing = child[1].white_check_1 >= 0 and player == 0 or child[1].black_check_1 >= 0 and player == 1
                if i_am_forcing:
                    extend = 1
                    counters[5] += i_am_forcing
            if i_am_forced:
                extend = 1
                counters[4] += i_am_forced
        child_extension = min(extension + extend, maxdepth - mindepth)

        move_value = -negamax_value(child, 1 - player, [x for x in available if x != move],
                                    summarizer, evaluator, table, counters, mindepth, maxdepth, -beta, -alpha, force,
                                    depth + 1, child_extension)

        # Update the record with the best value.
        # It is important to only accept the first best move because the later ones may
        # be only lower bounds due to pruning. This is because we use a weak beta cutoff,
        # which enables us to handle coarser evaluation functions.
        if move_value > best_value:
            best_value = move_value
            best_move = move
            best_child = child

        alpha = max(alpha, best_value) # This is the best score I am assured of working through the tree
        if alpha >= beta:
            counters[2] += 1
            break # Prune the branch. This means we will return a lower bound and not an exact value

    # Debugging of movealyzer changes
#    if i_am_forced and best_move != force_move:
#        force_child = [x for x in children if x[1] == force_move][0]
#        best_child =  [x for x in children if x[1] == best_move][0]
#        print('Mismatched force {} vs best {}'.format(pos2str(force_move), pos2str(best_move)))
#        print('Values:{} {}'.format(force_child[0], best_value))
#        print('force_child:', force_child[2][:-1])
#        print('best_child:', best_child[2][:-1])
#        print_board(board)
#        print(WHITE_OR_BLACK[player], 'to move')

#        raise ValueError('Mismatched force.')

    # Sanity check
    if abs(heuristic) >= WIN_THRESHOLD and abs(best_value) < WIN_THRESHOLD:
        raise ValueError('Heuristic is definite win or loss but value is not')

    # Fetch the new child information from the table
    best_child = unpack_record(best_child[-1])
#    if best_child[6] > 20:
#        raise ValueError('depth searched was {}'.format(best_child[6]))

    # Record the results in the transposition table
    buffer = record[-1]
    pack_values(buffer,
                heuristic, # The heuristic is included here for alignment reasons
                best_value,
                (UPPER_BOUND if best_value <= original_alpha else LOWER_BOUND if best_value >= beta else
                EXACT_VALUE),
                best_move, # The move corresponding to the best value
                required_height,
                best_child[7] + 1, # Update the maximum depth we searched to.
                extension,
                0
                )

    return best_value


##############################################
# A basic algorithm with no skill. This ultimately
# is equivalent to the random player.
##############################################

def basic_no_skill(record, player, allowed, summarizer, evaluator, table, counters, mindepth, maxdepth, force, seed):
    """A basic skill-less player for testing. The interface matches that of the negamax algorithm though most arguments are not used.
    The parameters that are used are record, player, allowed. Using this player should give play equivalent to random play.
    """
    allowed = sorted(allowed)
    board = record[0]
    values = [winlosedraw(move_board(board, m, player)) for m in allowed]
    wins = [m for m, v in zip(allowed, values) if v == player]
    loses = [m for m, v in zip(allowed, values) if v == 1 - player]
    draws = [m for m, v in zip(allowed, values) if v == DRAW]
    choices = [m for m, v in zip(allowed, values) if v == GAME_NOT_CONCLUDED]
    best_moves = wins or choices or draws or loses
    move = choice(best_moves, 0, seed + bytes(repr(board), 'utf-8'))
    return [move]

def null_summarizer(board, parent, player, move):
    return YavaFeatures1(0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0)


def null_evaluator(board, features):
    return 0

##############################################
# Some features classes, summarizers and evaluators
##############################################

def summarizer1(board, parent, player, move):
    white, black = board
    occupied = white | black
    white_threes = sum(1 for x in ALL_THREES if x & white == x)
    white_fours = sum(1 for x in ALL_FOURS if x & white == x)
    white_checks = set(p for w, e, p in ALL_CHECKS if w & white == w and e & occupied == 0)
    white_opens = sum(1 for w, e in ALL_OPENS if w & white == w and e & occupied == 0)
    white_traps = sum(1 for w, b, e in ALL_TRAPS if w & white == w and b & black == b and e & occupied == 0)

    black_threes = sum(1 for x in ALL_THREES if x & black == x)
    black_fours = sum(1 for x in ALL_FOURS if x & black == x)
    black_checks = set(p for b, e, p in ALL_CHECKS if b & black == b and e & occupied == 0)
    black_opens = sum(1 for b, e in ALL_OPENS if b & black == b and e & occupied == 0)
    black_traps = sum(1 for b, w, e in ALL_TRAPS if w & white == w and b & black == b and e & occupied == 0)

    return YavaFeatures1(white_threes=white_threes,
                         white_fours=white_fours,
                         white_check_1=max(white_checks) if white_checks else -1,
                         white_check_2=min(white_checks) if len(white_checks) > 1 else -1,
                         white_opens=white_opens,
                         white_traps=white_traps,
                         black_threes=black_threes,
                         black_fours=black_fours,
                         black_check_1=max(black_checks) if black_checks else -1,
                         black_check_2=min(black_checks) if len(black_checks) > 1 else -1,
                         black_opens=black_opens,
                         black_traps=black_traps,
                         unused_12=0, unused_13=0, unused_14=0, unused_15=0)


def summarizer2(board, parent, player, move):
    # This is an incrementally updating version of summarizer1
    if parent is None or parent[1] is None:
        return summarizer1(board, parent, player, move)
    # Incrementally update
    features = parent[1]
    white, black = board
    occupied = white | black
    pwhite, pblack = parent[0]
    poccupied = pwhite | pblack
    if player == 0:
        white_threes = features.white_threes + sum(1 for x in THREES_BY_POSITION[move] if x & white == x)
        white_fours = features.white_fours + sum(1 for x in FOURS_BY_POSITION[move] if x & white == x)
        black_threes = features.black_threes
        black_fours = features.black_fours
    else:
        black_threes = features.black_threes + sum(1 for x in THREES_BY_POSITION[move] if x & black == x)
        black_fours = features.black_fours + sum(1 for x in FOURS_BY_POSITION[move] if x & black == x)
        white_threes = features.white_threes
        white_fours = features.white_fours

    add_white_checks = set(p for w, e, p in CHECKS_BY_POSITION[move] if w & white == w and e & occupied == 0)
    del_white_checks = set(p for w, e, p in CHECKS_BY_POSITION[move] if w & pwhite == w and e & poccupied == 0)
    old_white_checks = set(x for x in (features.white_check_1, features.white_check_2) if x != -1)
    white_checks = (old_white_checks | add_white_checks) - del_white_checks

    #white_checks = (features.white_checks + sum(1 for w, e in CHECKS_BY_POSITION[move] if w & white == w and e & occupied == 0)
    #                - sum(1 for w, e in CHECKS_BY_POSITION[move] if w & pwhite == w and e & poccupied == 0))

    white_opens = (features.white_opens
                   + sum(1 for w, e in OPENS_BY_POSITION[move] if w & white == w and e & occupied == 0)
                   - sum(1 for w, e in OPENS_BY_POSITION[move] if w & pwhite == w and e & poccupied == 0))
    white_traps = (features.white_traps
                   + sum(1 for w, b, e in TRAPS_BY_POSITION[move] if w & white == w and b & black == b and e & occupied == 0)
                   - sum(1 for w, b, e in TRAPS_BY_POSITION[move] if w & pwhite == w and b & pblack == b and e & poccupied == 0))

    add_black_checks = set(p for b, e, p in CHECKS_BY_POSITION[move] if b & black == b and e & occupied == 0)
    del_black_checks = set(p for b, e, p in CHECKS_BY_POSITION[move] if b & pblack == b and e & poccupied == 0)
    old_black_checks = set(x for x in (features.black_check_1, features.black_check_2) if x != -1)
    black_checks = (old_black_checks | add_black_checks) - del_black_checks

#    black_checks = (features.black_checks + sum(1 for b, e in CHECKS_BY_POSITION[move] if b & black == b and e & occupied == 0)
#                    - sum(1 for b, e in CHECKS_BY_POSITION[move] if b & pblack == b and e & poccupied == 0))

    black_opens = (features.black_opens
                   + sum(1 for b, e in OPENS_BY_POSITION[move] if b & black == b and e & occupied == 0)
                   - sum(1 for b, e in OPENS_BY_POSITION[move] if b & pblack == b and e & poccupied == 0))
    black_traps = (features.black_traps
                   + sum(1 for b, w, e in TRAPS_BY_POSITION[move] if w & white == w and b & black == b and e & occupied == 0)
                   - sum(1 for b, w, e in TRAPS_BY_POSITION[move] if white & pwhite == w and b & pblack == b and e & poccupied == 0))


    new_features = YavaFeatures1(white_threes=white_threes,
                                 white_fours=white_fours,
                                 white_check_1=min(white_checks) if white_checks else -1,
                                 white_check_2=max(white_checks) if len(white_checks) > 1 else -1,
                                 white_opens=white_opens,
                                 white_traps=white_traps,
                                 black_threes=black_threes,
                                 black_fours=black_fours,
                                 black_check_1=min(black_checks) if black_checks else -1,
                                 black_check_2=max(black_checks) if len(black_checks) > 1 else -1,
                                 black_opens=black_opens,
                                 black_traps=black_traps,
                                 unused_12=0, unused_13=0, unused_14=0, unused_15=0)


    # test = summarizer1(board, parent, player, move)
    # if test != new_features:
    #     print('test:', test)
    #     print('new_features:', new_features)
    #     print('move:', move)
    #     print('player:', move)
    #     print('board:', board)
    #     print('parent features:', features)
    #     print('parent board:', parent[1])
    #     print_board(board)
    #     raise ValueError('Incorrect feature calculation')
    return new_features


def evaluator1(board, features, check=2000, open=1000, trap=0, depth=1, obvious=0):
    white, black = board
    # Calculate the player to move from the board depth
    board_depth = bin(white | black).count('1') # This counts the number of occupied positions
    player = board_depth % 2

    white_checks = 0 if features.white_check_1 < 0 else 1 if features.white_check_2 < 0 else 2
    black_checks = 0 if features.black_check_1 < 0 else 1 if features.black_check_2 < 0 else 2

    # Evaluate for white
    score = None
    if features.white_fours > 0 or (features.black_threes > 0 and not features.black_fours > 0):
        score = WIN_VALUE # White wins
    elif features.black_fours > 0 or (features.white_threes > 0 and not features.white_fours > 0):
        score = LOSE_VALUE # Black wins
    elif white | black == FULL:
        score = 0
    elif obvious:
        # Score some "obvious" wins and losses. This helps add an extra ply for weak players
        if white_checks > 0 and player == 0:
            score = obvious - depth # Win for white on next move
        elif white_checks > 1 and black_checks == 0:
            score = obvious - 2 * depth# Win for white in 2 moves
        elif black_checks > 0 and player == 1:
            score = obvious + depth # Win for black on next move
        elif black_checks > 1 and white_checks == 0:
            score = obvious + 2 * depth # Win for black on next 2 moves
    # Score the cases that are not clear wins or losses
    if score is None:
        white_score = check * white_checks + open * features.white_opens + trap * features.white_traps
        black_score = check * black_checks + open * features.black_opens + trap * features.black_traps
        score = white_score - black_score

    # Add a term for depth so that earlier wins count more and later loses count more
    # Good positions near the start of the game get a bonus and bad positions near the end of the game get a penalty.
    score = score + depth * (61 - board_depth) if score > 0 else score - depth * (61 - board_depth) if score < 0 else score


    return (1 - 2 * player) * score



######################################################
# A human player
######################################################

class YavaInteractive1(object):
    """A Yavalath player object that reads moves from user input
    """

    def __init__(self, ):
        # Properties that vary from game to game but are fixed through the game
        self.player = None

        # A record of the game and other player state
        self.game_moves = None
        self.current = None
        self.allowed = None


    def start_game(self, player):
        print('Starting new game:')
        print('You are ', WHITE_OR_BLACK[player])
        self.player = player
        self.game_moves = []
        self.game_stats = []
        self.current = EMPTY_BOARD
        self.allowed = set(range(61))


    def update(self, move):
        # Ignore a repeated message. This helps the player handle different protocols more easily.
        if self.game_moves and move == self.game_moves[-1]:
            return
        # Do the update
        player = len(self.game_moves) % 2 # Player 0 is white
        self.game_moves.append(move)
        posmove = str2pos(move)
        bitmove = str2bits(move)
        self.current = move_board(self.current, posmove, player)
#        print('Removing move from allowed set', posmove)
        self.allowed = self.allowed - {posmove}


    def your_move(self):
        available = sorted(self.allowed)
        strmove = None
        while strmove is None:
            inmove = input('Your move:')
            if inmove not in STR_2_POS:
                print('{} is not a valid move')
                continue
            posmove = STR_2_POS[inmove]
            if posmove not in self.allowed:
                print('Move {} has already been taken and is not allowed'.format(strmove))
                continue
            strmove = inmove
        self.update(strmove)
        return strmove

    def get_stats():
        return self.game_stats


######################################################
# Factory methods for constructing predefined players
######################################################

def make_player(name, verbose=False):
    players = { 'adam':  lambda: YavaRandom1(b'7gf872gfw3od7jftfoc7y8456296', verbose=verbose),
                'arael': lambda: YavaNega1(null_summarizer, null_evaluator, basic_no_skill, 0, verbose=verbose),
                'armisael': lambda: YavaNega1(summarizer2, evaluator1, negamax, mindepth=4, deepen=False, verbose=verbose),
                'bardiel': lambda: YavaNega1(summarizer2, lambda b, f: evaluator1(b, f, trap=500, obvious=2000000),
                                             negamax, mindepth=5, maxdepth=15, deepen=False, force=True, verbose=verbose),
                'gaghiel': lambda: YavaNega1(summarizer2, lambda b, f: evaluator1(b, f, trap=500, obvious=2000000),
                                             negamax, mindepth=3, maxdepth=10, deepen=False, force=True, verbose=verbose),
                'ireul': lambda: YavaNega1(summarizer2, lambda b, f: evaluator1(b, f, trap=500, obvious=2000000),
                                             negamax, mindepth=6, maxdepth=18, deepen=False, force=True, verbose=verbose),
                'interactive': lambda: YavaInteractive1()
                }
    return players[name]()


def get_player(name, verbose=False):
    return CommonInterface(make_player(name, verbose))


def get_player_names():
    return 'armisael bardiel gaghiel ireul'.split()


#######################################################
# Tournament and play harnesses
#######################################################

def play_game(white, black, white_name=None, black_name=None, print_game=False, verbose=False, initial=(),
              dots=False):
    white_name = white_name if white_name is not None else white if isinstance(white, str) else 'anonymous'
    black_name = black_name if black_name is not None else black if isinstance(black, str) else 'anonymous'
    players_names = white_name, black_name
    # Initialize the players on separate lines to make debuggin easier
    white_player = make_player(white, verbose) if isinstance(white, str) else white
    black_player = make_player(black, verbose) if isinstance(black, str) else black
    players = white_player, black_player
    # Set up an initial game which by default is empty
    game_moves = list(initial)
    game_stats = [] # Times for move/update and update for each move and player
    board = EMPTY_BOARD
    # Inform the players we are about to start a game and tell them which color they are playing.
    for i, player in enumerate(players):
        player.start_game(i % 2)
    # Update the players with the initial portion of the game
    for turn, move in enumerate(initial):
        player_to_move = turn % 2
        move_pos = str2pos(move)
        board = move_board(board, move_pos, player_to_move)
        for player in players:
            player.update(move)
    # print an initial board if requested
    if print_game:
        print_board(game_moves)
        print()
    # Run the game
    for turn in range(len(initial), 61):
        player_to_move = turn % 2
        other_player = (turn + 1) % 2
        # Request a move
        if print_game:
            print('Move {}: {} ({}) to move'.format(turn, WHITE_OR_BLACK[player_to_move], players_names[player_to_move]),
                  flush=True)
        start = time.time()
        move = players[player_to_move].your_move()
        end = time.time()
        move_time = end - start
        if print_game:
            print("Move {}: {} ({}) moves {}".format(turn, WHITE_OR_BLACK[player_to_move], players_names[player_to_move], move))
            print("Time taken for move:", move_time)
        if dots:
            print('.', end='', flush=True)
        # Test the move for validity
        move_pos = str2pos(move)
        if not valid_move(board, move_pos):
            raise ValueError('{} is not a valid move for this board'.format(move))
        # Another check for paranoia
        if move in game_moves:
            raise ValueError('Move {} has already been played'.format(move))
        # Apply the move to the game
        game_moves.append(move)
        board = move_board(board, move_pos, player_to_move)
        if print_game:
            print('Board after move {}:'.format(turn))
            print()
            print_board(board, move)
            print()
            print('Game so far: {}'.format(' '.join(game_moves)))
            print(flush=True)
        # Inform the other player of the move. We do this regardless of whether the
        # game is over, so that the player could gather it's own statistics
        start = time.time()
        players[other_player].update(move)
        end = time.time()
        update_time = end - start
        # Update the game stats
        game_stats.append((move_time, update_time) if player_to_move == 0 else (update_time, move_time))
        # Test for wins, losses, and draws
        value = winlosedraw(board)
        if value != GAME_NOT_CONCLUDED:
            break
    return value, game_moves, game_stats


def print_game_summary(game_result):
    print('Game outcome:', result_string(game_result[0]))
    print('Game moves: {}'.format(' '.join(game_result[1])))
#    print('Total white time:', sum(x[0] for x in game_result[2]))
#    print('Total black time:', sum(x[0] for x in game_result[2]))
    print('Game final board:')
    print_board(game_result[1], game_result[1][-1])
    print(flush=True)


def tournament(players, initial_moves):
    """Run a tournament between several players that compares all players with each other
    The tournament also allows the games to be started from different positions.

    Args:
        players:
            list(str): The names of the players in the tournament.

    initial_moves:
            list(list(str)). A list of initial games. E.g., call unfinished_games(1)
            to get all 9 games with one move up to symmetry.

    Returns:
        A list of tuples. (white_player, black_player, initial_moves, result, game)
    """
    games = []
    for i, player1 in enumerate(players):
        for j, player2 in enumerate(players):
            if i == j:
                continue
            for initial in initial_moves:
                print('{} (white) vs {} (black):'.format(player1, player2))
                print('Initial moves:', ' '.join(initial))
#                print_board(initial)
                game = play_game(player1, player2, print_game=False, verbose=False, initial=initial)
                print()
                print_game_summary(game)
                games.append((player1, player2, initial, game[0], game[1]))
    return games

######################################################
# An interface so that the players in this module can
# be plugged into other systems
######################################################

class CommonInterface(object):

    def __init__(self, player):
        self.player = player
        self.game_so_far = None

    def __call__(self, game_so_far):
        self.move(game_so_far)

    def move(self, game_so_far):
        game_so_far = [x.lower() for x in game_so_far]
        if not game_so_far or game_so_far[:-1] !=  self.game_so_far:
            player_number = len(game_so_far) % 2
            self.player.start_game(player_number)
            for move in game_so_far:
                self.player.update(move)
            self.game_so_far = game_so_far
        else:
            # This is an update to an old game so tell the player about the last move
            self.player.update(game_so_far[-1])


        # Figure out a move to play and record it in the local copy of the game
        move = self.player.your_move()
        if move is not None:
            self.game_so_far.append(move)
        # Return the move
        return move

    def get_stats():
        return self.player.get_stats()


##########################################################
# Testing and QA
##########################################################

def analyze_one_move(player, game):
    game = game.split() if isinstance(game, str) else list(game)
    player_number = len(game) % 2
    player.start_game(player_number)
    for move in game:
        player.update(move)
    move = player.your_move()
    return move

def validate_game(game, engines, names, verbose=False):
    """This validates a game against recalculations of one or more players.

    Args:
        game:
            list(str). A listing of the games moves until the last move that was played.

        engines:
            A pair of game players engines with a move method. The move method accepts the game
            up to a point and returns the move. If the entry for a player is None, then
            the moves for that player will be read from the game

        names:
            A pair of names that are the names of the players

        verbose:
            Set to True to print a move by move replay and comparison of the game
            and the recalculated game.

    Returns:
        True if the recalculated game matches the game passed in. False if they do not match.
    """
    printv = lambda *args, **kwargs: print(*args, **kwargs) if verbose else None
    print_boardv = lambda *args, **kwargs: print_board(*args, **kwargs) if verbose else None
    printv('Validating the game {} vs {}'.format(*names))
    game = list(game) # Ensure that we have list, so that game comparisons work later
    recalculated_game = []
    for turn, game_move in enumerate(game):
        player = turn % 2
        name = names[player]
        printv('----- Turn {}, {} to move -----'.format(turn, name))
        if engines[player] is None:
            printv('Reading move for {} from the game log'.format(name))
            move = game_move
            printv('{} moves: {}'.format(name, move))
        else:
            printv('Recalculating move for {}'.format(name))
            move = engines[player].move(game[:turn])
            printv('{} moves: {}'.format(name, move))
            # Validation step
            if move == game_move:
                printv('Move for {} matches the move from the game log'.format(name))
            else:
                printv('MOVE MISMATCH: Move {} for {} does not match move {} from the game log'.format(move, name, game_move))
        recalculated_game.append(move)
        printv('Board for recalculated game after turn {}:'.format(turn))
        print_boardv(recalculated_game[:turn + 1], recalculated_game[turn])
        if game[:turn + 1] != recalculated_game[:turn +1]:
            printv('Board for game from the game log:')
            print_boardv(game[:turn + 1], recalculated_game[turn])
    if game == recalculated_game:
        printv('Recalculated game matches the game log')
        return True
    printv('Recalculated game does not match the game log')
    printv('game:', ' '.join(game))
    printv('recalculated game:', ' '.join(recalculated_game))
    printv('Final board for game:')
    print_boardv(game, game[-1])
    printv('Final board for recalculated game:')
    print_boardv(recalculated_game, recalculated_game[-1])
    return False


def explore_table(table, start, show=True):
    game = list(start)
    board = sboard2bboard(game)
    key = BoardStruct.pack(*board)
    if key not in table:
        print('Starting board not in table:', ' '.join(game))
        return
    while True:
        # Calculation section
        board = sboard2bboard(game)
        key = BoardStruct.pack(*board)
        record = unpack_record(table[key])
        white, black = board
        occupied = white | black
        available = [x for x in range(61) if (1 << x) & occupied == 0]
        available = sorted(available, key=pos2str)
        child_boards = [move_board(board, x, player_to_move(board)) for x in available]
        child_keys = [BoardStruct.pack(*child) for child in child_boards]
        table_moves = [move for move, child_key in zip(available, child_keys) if child_key in table]
        child_records = [unpack_record(table[child_key]) for move, child_key in zip(available, child_keys) if child_key in table]
        child_values = sorted([(child[3], child[2], pos2str(move), child[4]) for move, child in zip(table_moves, child_records)])
        strmoves = [pos2str(x) for x in table_moves]
        # Display section
        if show:
            # Verbose display
            print_board(game)
            print('Player to move:', WHITE_OR_BLACK[player_to_move(board)])
            print('Game:', ' '.join(game))
            print('Board:', board)
            print('Value:', record[3])
            print('Heuristic:', record[2])
            print('Accuracy:', record[4])
            print('Min search depth:', record[5])
            print('Max search depth:', record[6])
            print('Min best depth:', record[7])
            print('Max best depth:', record[8])
            print('Features:', record[1])
            print('Available:', ' '.join(pos2str(x) for x in available))
            print('Moves in table:', ' '.join(pos2str(x) for x in table_moves))
            print('Summary of children in table:')
            print('Child value heuristic accuracy')
            for value, heuristic, child, accuracy in child_values:
                print('{}    {}  {} {}'.format(child, value, heuristic, accuracy))
        else:
            print_board(game)
            print('Player to move:', WHITE_OR_BLACK[player_to_move(board)])
            print('Game:', ' '.join(game))
            print('Moves in table:', ' '.join(pos2str(x) for x in table_moves))
        # Command section
        while True:
            command = input('Command (up/quit/show or a move):')
            if command == 'quit':
                return
            if command == 'show':
                show = True
                break
            if command == 'hide':
                show = False
                break
            if command == 'up':
                updated = game[:-1]
            elif command in strmoves:
                updated = game + [command]
            else:
                print('{} is not a valid command'.format(command))
                continue
            newboard = sboard2bboard(updated)
            newkey = BoardStruct.pack(*newboard)
            if newkey not in table:
                print('The board for your command is not in the table')
                continue
            game = updated
            break

# Some games that Franco sent over
# White: Classy ID Player depth 3, branching 61, use_mc False
# Black: Bardiel
# 0 Winner: Black By Illegal Move. White/Black Time: 27.7/3.4 History: ['e5', 'd4', 'h5', 'f5', 'e8', 'f7', 'e6', 'e7']
# 1 Winner: Black. White/Black Time: 33.8/5.1 History: ['e6', 'e4', 'h6', 'f6', 'e9', 'f8', 'b6', 'f5', 'b4', 'f7']
# 2 Winner: Black. White/Black Time: 31.8/4.7 History: ['e7', 'e4', 'g5', 'f6', 'd7', 'd5', 'd4', 'c4', 'a4', 'e6']
# 3 Winner: Black By Illegal Move. White/Black Time: 40.3/4.7 History: ['e8', 'e5', 'b2', 'b3', 'e2', 'c2', 'h5', 'f7']
# 4 Winner: Black By Illegal Move. White/Black Time: 37.8/4.7 History: ['e9', 'e6', 'b6', 'c7', 'h3', 'h4', 'e3', 'f3']
# 5 Winner: Black By Illegal Move. White/Black Time: 26.6/3.4 History: ['f5', 'd4', 'i5', 'g5', 'i2', 'i3', 'g4', 'h3']
# 6 Winner: Black. White/Black Time: 30.8/4.7 History: ['f6', 'd4', 'b1', 'f5', 'a3', 'c3', 'e5', 'a1', 'e1', 'b2']
# 7 Winner: Black. White/Black Time: 26.4/7.5 History: ['f7', 'f4', 'f3', 'd3', 'c1', 'g4', 'e4', 'i4', 'c4', 'h4']
# 8 Winner: Black. White/Black Time: 27.2/9.1 History: ['f8', 'f5', 'e9', 'e7', 'e6', 'd6', 'f7', 'b4', 'c6', 'c5']
# 9 Winner: Black By Illegal Move. White/Black Time: 24.8/4.6 History: ['g6', 'd5', 'f7', 'f4', 'i4', 'h5', 'd8', 'e8']
#  Show original message

SG_FB_20170604_00 = ['e5', 'd4', 'h5', 'f5', 'e8', 'f7', 'e6', 'e7']
SG_FB_20170604_01 = ['e6', 'e4', 'h6', 'f6', 'e9', 'f8', 'b6', 'f5', 'b4', 'f7']
SG_FB_20170604_02 = ['e7', 'e4', 'g5', 'f6', 'd7', 'd5', 'd4', 'c4', 'a4', 'e6']
SG_FB_20170604_03 = ['e8', 'e5', 'b2', 'b3', 'e2', 'c2', 'h5', 'f7']
SG_FB_20170604_04 = ['e9', 'e6', 'b6', 'c7', 'h3', 'h4', 'e3', 'f3']
SG_FB_20170604_05 = ['f5', 'd4', 'i5', 'g5', 'i2', 'i3', 'g4', 'h3']
SG_FB_20170604_06 = ['f6', 'd4', 'b1', 'f5', 'a3', 'c3', 'e5', 'a1', 'e1', 'b2']
SG_FB_20170604_07 = ['f7', 'f4', 'f3', 'd3', 'c1', 'g4', 'e4', 'i4', 'c4', 'h4']
SG_FB_20170604_08 = ['f8', 'f5', 'e9', 'e7', 'e6', 'd6', 'f7', 'b4', 'c6', 'c5']
SG_FB_20170604_09 = ['g6', 'd5', 'f7', 'f4', 'i4', 'h5', 'd8', 'e8']

SB_AR_GA_20170617_01 = """
        1 2 3 4 5
      a - - - - - 6
    b X - - x - - 7
    c - - - o o - - 8
  d - - - - x - - - 9
  e x x o x o x - o - e
  f o - - o - o - - 9
    g - - - - - x - 8
    h - - - - - - 7
      i - - - - - 6
        1 2 3 4 5
"""
SB_AR_GA_20170617_01 = 'e5 e4 c5 g6 f1 e1 f4 d5 e8 b4 c4 e6 f6 e2 e3 b1'.split()


#######################################################
# Testing on the iPad
#######################################################

def test_ipad_games(verbose=False):
    game_1 = play_game('adam', 'arael', print_game=verbose)
    if verbose:
        print_game_summary(game_1)

def all_ipad_tests(verbose=False):
    test_ipad_games(verbose=True)

def interactive_play():
    for i in range(10):
        print('----------------------')
        print('Game {}'.format(i))
        players = ('interactive', 'gaghiel') if i % 2 == 0 else ('gaghiel', 'interactive')
        game = play_game('interactive', 'gaghiel', print_game=True, verbose=True)
        print_game_summary(game)
        print('----------------------')
    print('GAME OVER')

if __name__ == '__main__':
    pass
#    interactive_play()

#    print(unfinished_games(1))
#    tournament(['armisael', 'gaghiel'], unfinished_games(1))

#    games = tournament(['armisael', 'gaghiel', 'bardiel'], unfinished_games(1))
#    print(games)
#    print(unfinished_games(2))

#    for game in unfinished_games(2):
#        player = make_player('ireul', verbose=True)
#        game = list(game)
#        print('---------------------', flush=True)
#        print('Initial:', ' '.join(game))
#        print_board(game)
#        move = analyze_one_move(player, game)
#        print('Principal variation:', ' '.join(player.principal_variation))
#        print('Result: {}: {}'.format(result_string(winlosedraw(sboard2bboard(game + player.principal_variation))),
#                                      ' '.join(game + player.principal_variation)))
#        print_board(game + player.principal_variation, game)


#    game = play_game('gaghiel', 'bardiel', print_game=True, verbose=True)
#    print()
#    print_game_summary(game)


#    evaluator1_1 = lambda board, features: evaluator1(board, features, check=2000, open=1000, depth=1, obvious=200000)

#    armisael = make_player('armisael', verbose=True)
#    bardiel = make_player('bardiel', verbose=True)
#    gaghiel = make_player('gaghiel', verbose=True)


#    armisael_1 = YavaNega1(summarizer2, evaluator1_1, negamax, mindepth=4, deepen=False, verbose=True, clean=True)
#    gaghiel_1 = YavaNega1(summarizer2, lambda b, f: evaluator1(b, f, check=1000, open=2000, obvious=2000000),
#                          negamax, mindepth=3, maxdepth=10, deepen=False, force=True, verbose=True)
#
#    game = play_game(gaghiel_1, 'gaghiel', 'gaghiel_1', 'gaghiel', print_game=True, verbose=True)
#    print()
#    print_game_summary(game)

#    game = play_game(make_player('gaghiel', verbose=True), 'armisael', 'gaghiel', print_game=False, verbose=True, initial=['e5'])
#    print()
#    print_game_summary(game)

    # A problem move for gaghiel. A forcing bug
#    game = 'e5 e4 c5 g6 f1 e1 f4 d5 e8 b4 c4 e6 f6 e2 e3 b1'.split()
#    armisael = make_player('armisael', verbose=True)
#    armisael = YavaNega1(summarizer2, evaluator1, negamax, mindepth=4, deepen=False, verbose=True, clean=False)
#    gaghiel = YavaNega1(summarizer2, lambda board, features: evaluator1(board, features, check=1000, open=2000, depth=1, obvious=200000), negama#x, mindepth=3, maxdepth=10, deepen=False, force=True, verbose=True, clean=False)
#    print('---------------------')
#    print_board(game)
#    move = analyze_one_move(gaghiel, game)
#    print('Move:', move)
#    print_board(game + [move], move)
#    print('\nINTERACTIVE EXPLORATION\n')
#    explore_table(armisael.table, game)

#    import cProfile
#    cProfile.run("play_game('armisael', 'arael', print_game=True, verbose=True)")
#    length = 2
#    games = list_game_boards(length)
#    print('There are {} boards for games of length {}'.format(len(games), length))
#    for game in games:
#        print(' '.join(game[1]), result_string(game[2]))

    #game_to_check = ['d7', 'd4', 'c7', 'd2', 'f6', 'e7', 'f3', 'f5', 'c4', 'c3', 'e5', 'a1', 'b2', 'd1', 'd3', 'b1', 'c1', 'c6', 'e6', 'd5', 'i3', 'h3', 'h2', 'g3', 'e3', 'e4', 'h6', 'g6', 'g4', 'g5', 'i5', 'f8', 'g1', 'f2', 'h5', 'c5', 'a4', 'i4', 'b5', 'b3', 'h1', 'd8', 'f7', 'a5', 'e2', 'f4', 'e8', 'c2', 'a3'] # Note the last move did not occur in the tournament
    #armisael = get_player('armisael', verbose=True)
    #out = validate_game(game_to_check, (armisael, None), ('armisael', 'bologna'), verbose=True)
    #print(out)

#    armisael = make_player('armisael', verbose=True)
#    armisael.start_game(0)
#    for move in game_to_check[:-1]:
#        armisael.update(move)
#    move = armisael.your_move()
#    print(move)

    #print(pos2str(18))
    #all_ipad_tests()

#    board = EMPTY_BOARD
#    move = 1
#    player = 1
#    board = move_board(board, move, player)
#    print_board(board, move)

#    print('Game 1:')
#    game1_players = ('dkmyavarandom1.1', 'dkmyavarandom1.2')
#    game1 = play_game(game1_players[0], game1_players[1], print_game=True)
#    print('Game 1 result:', result_string(game1[0]))
#    print('Game 1 moves: {}'.format(' '.join(game1[1])))
#    print('Game 1 final board:')
#    print_board(game1[1], game1[1][-1])

#    print(len(set(STRING_POSITIONS)))
#    sboard = 'e5 a5 b2 a1 i5 f7'.split()
#    print_board(sboard)
#    print()
#    bboard = sboard2bboard(sboard)
#    print_board(bboard)
#    print('flip 0:')
#    print_board(flip(bboard, 0))
#    print('flip 1:')
#    print_board(flip(bboard, 1))
#    print('rotate 1')
#    print_board(rotate(bboard, 1))
#  print()
#    print(' '.join(SOME_THREES))
#    print(' '.join(bits2str(y) for y in sorted(set(to_fundamental_domain(str2bits(x))[0] for x in SOME_THREES))))
#    print('Threes:', ' '.join(bits2str(x) for x in ALL_THREES))
#    print('Fours:', ' '.join(bits2str(x) for x in ALL_FOURS))
#    print('Wide fours:', ' '.join(bits2str(x) for x in ALL_WIDE_FOURS))
#    print(len(ALL_THREES), len(ALL_FOURS), len(ALL_SPLIT_FOURS), len(ALL_WIDE_FOURS))

#    print(hashlib.algorithms_available)
#    h = hashlib.new('sha1')
#    h.update(bytes(5))
#    print(struct.unpack('l', h.digest()[:8])[0] % 17)

#    bboard = sboard2bboard('a1 e5 f3 e6 f4 e8 b2 e7'.split())
#    print_board(bboard)
#    print(winlosedraw(bboard))
#    bboard = bboard[0], bboard[1]
#    print_board(bboard)
#    print(winlosedraw(bboard))
#    print_board(bboard, marked=1)
#    print_board(bboard, marked='e5e6')
#    print_board(bboard, marked='e5 e6 e7'.split())
#    print_board(bboard, marked=[0,1,2,3])
#    print_board((0, FULL), marked=0)
#    print_board((0, FULL), marked=FULL)

# import dkmyava1
# from dkmyava1 import (YavaNega1, summarizer2, evaluator1, negamax1, print_game_summary, print_board,
#                      null_summarizer, null_evaluator, basic_no_skill, CommonInterface)

# player_1 = YavaNega1(summarizer2, evaluator1, negamax1, mindepth=3)
# player_2 = YavaNega1(summarizer2, evaluator1, negamax1, mindepth=4, verbose=True, clean=False)
# player_3 = YavaNega1(summarizer2, lambda b, f, p: evaluator1(b, f, p, 100, 30), negamax1, mindepth=4, verbose=True, clean=False)

# player_2 = YavaNega1(summarizer2, evaluator1, negamax1, mindepth=4, verbose=True, clean=False, prune=False)
# player_3 = YavaNega1(summarizer2, lambda b, f, p: evaluator1(b, f, p, 100, 30), negamax1, mindepth=4, verbose=True,
#                      clean=False, prune=False)


# player_4 = YavaNega1(null_summarizer, null_evaluator, basic_no_skill, 0)
# player_5 = YavaNega1(summarizer2, lambda b, f, p: evaluator1(b, f, p, 100, 20), negamax1, mindepth=4)
# player_6 = YavaNega1(summarizer2, lambda b, f, p: evaluator1(b, f, p, 50, 20), negamax1, mindepth=4)
# player_7 = YavaNega1(summarizer2, evaluator1, negamax1, mindepth=4, force=True, verbose=True) force=True
# player_8 = YavaNega1(summarizer2, lambda b, f, p: evaluator1(b, f, p, 100, 20), negamax1, mindepth=4, force=True)
# player_9 = YavaNega1(summarizer2, evaluator1, negamax1, mindepth=5, force=True, verbose=True) force=True
# player_10 = YavaNega1(summarizer2, evaluator1, negamax1, mindepth=5, verbose=True) force=True

# player_11 = YavaNega1(summarizer2, evaluator1, negamax1, mindepth=2, verbose=True)
# player_12 = YavaNega1(summarizer2, evaluator1, negamax1, mindepth=2, verbose=True)

# game_3_2 = dkmyava1.play_game(player_3, player_2, True, 'player_3', 'player_2')
# print_game_summary(game_3_2)