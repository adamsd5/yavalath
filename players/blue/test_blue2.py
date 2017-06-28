import unittest
import yavalath_engine
from players.blue.blue2 import player
import pprint
import collections
import pickle
import timeit
import numpy
import pathlib
import itertools

class TestBlue2Player(unittest.TestCase):

    def test_breathing(self):
        game_so_far = ['a1']
        player(game_so_far, depth=2, verbose=True)

    def test_self_trap_1(self):
        # Groucho vs. Abbot, A shorter version where groucho traps himself with the check at f2
        game_so_far = ['c4', 'd8', 'c1', 'g7', 'c2', 'c3', 'e2', 'h1', 'f1', 'd2', 'd3', 'a4', 'f4', 'e4'] # , 'f2', 'f3', 'g2'
        move = player(game_so_far, depth=1, verbose=False)
        print("Move selected:", move)
        self.assertNotEqual(move[0], 'f2')

    def test_chico_crash_1(self):
        # This is the same bug as the suicide-block...
        game_so_far = ['d5', 'g7', 'g5', 'e3', 'h3', 'i5', 'h2']
        yavalath_engine.Render(yavalath_engine.HexBoard(), game_so_far).render_image("test_chico_crash_1.png")
        cutoff = game_so_far + ['i4', 'h5', 'h4', 'e5', 'f5', 'g3', 'f4', 'g4', 'i2', 'i3', 'f2']
        yavalath_engine.Render(yavalath_engine.HexBoard(), cutoff).render_image("test_chico_crash_1_cutoff.png")

        game_so_far = game_so_far + ['i4', 'h5', 'h4', 'e5', 'f5', 'g3', 'f4']  # blue2 plays at 'g4' to block, but it is suicidal

        move = player(game_so_far, depth=2, verbose=False)
        self.assertNotEqual(move[0], 'f2')

    def test_force_logic_1(self):
        game_so_far = ['g2', 'c2', 'g7', 'i5', 'd1', 'e6', 'c4', 'a2', 'b3', 'e5', 'g4', 'g5', 'c1']
        move_stack = ['d2', 'b2', 'd4', 'f5', 'd5', 'd3', 'b5', 'c5', 'f1', 'e2', 'f2', 'a1', 'b1', 'c7', 'c6', 'e8', 'g6', 'f6', 'b6', 'd6', 'a3', 'g1']
        move = player(game_so_far, depth=1, verbose=True)
        print("Move:", move)



# In the following game, chico vs. abbot, chico trapped himself at the end.  The check (h2) move never should have been made,
# because it forces abbot to win.  This should have been caught with a deeper search in 'forced' mode.
game_so_far = ['f6', 'i5', 'a1', 'd3', 'b5', 'g3', 'a2', 'd2', 'a4', 'a3', 'd7', 'c6', 'c7', 'e7', 'e3', 'e2', 'b3', 'e8', 'b2', 'b4', 'd4', 'c3', 'd6', 'd5', 'f4', 'i2', 'f3', 'f5', 'g2', 'e4', 'f1', 'f2', 'i3', 'c4', 'c5', 'e1', 'h4', 'g5', 'h5', 'd8', 'h2', 'h3', 'g4']

# In this game, chico vs. groucho, chico wins, but should have won a move earlier.  Move 28 at d7 never should have been made.
game_so_far = ['g2', 'e1', 'g1', 'g4', 'b3', 'a5', 'i3', 'a3', 'a2', 'd5', 'b5', 'b2', 'b6', 'b4', 'd6', 'e6', 'g6', 'h3', 'f5', 'h2', 'h5', 'i5', 'e5', 'g5', 'e8', 'f7', 'c3', 'd4', 'd7', 'c6', 'e3', 'd3']

# Groucho vs. Abbot, A shorter version where groucho traps himself with the check at f2
game_so_far = ['c4', 'd8', 'c1', 'g7', 'c2', 'c3', 'e2', 'h1', 'f1', 'd2', 'd3', 'a4', 'f4', 'e4', 'f2', 'f3', 'g2']

# Grouch vs. chico.  Chico forces the right win here, but the check at c1 could have been skipped.
game_so_far = ['a4', 'e5', 'i4', 'c6', 'd6', 'd5', 'g3', 'b5', 'c5', 'e8', 'd7', 'e7', 'e6', 'g4', 'a3', 'b4', 'a1', 'a2', 'b2', 'c4', 'b3', 'e4', 'd4', 'c3', 'd3', 'e2', 'e3', 'h4', 'f4', 'c1', 'c2', 'g5', 'f6', 'g7', 'g6']

# Chico vs. groucho.  Groucho wins this.  Move 9 at b6 was the winning move.  Why didn't Chico see that?  Test this with 7 moves and see if players know to block at b6
game_so_far = ['h5', 'd8', 'e9', 'd6', 'd5', 'a2', 'g2', 'e6', 'g4', 'b6', 'c6', 'a5', 'c7', 'a3', 'a4', 'c5', 'b4', 'f7', 'e7', 'g6', 'e8']

# Grouch vs. Chico, Groucho wins.  Move 13 at a3 allowed moved 14 at f7 for the win.
game_so_far = ['b3', 'g7', 'e9', 'a4', 'f5', 'f8', 'i5', 'g5', 'g4', 'i2', 'd6', 'e6', 'c5', 'a3', 'f7', 'e7', 'f4', 'f6']

# Issue 1: If all checks lead to an eventual win, I should order them based on the shortest sequence to win.
# Issue 2: Soemthing is allowing Groucho to trap himself into a loss by checking.

# 0626_005616: Groucho vs. Chico, Groucho wins... chico makes a terrible move 21 ag g6, leaving h3 open, taken by Groucho at move 24 after toying a bit
# Note also that chico suicides to block... something he shouldn't ever do.
game_so_far = ['h1', 'h4', 'c6', 'd8', 'f6', 'c2', 'f3', 'g2', 'g3', 'f4', 'e4', 'd7', 'e7', 'e6', 'h6', 'f8', 'i4', 'h5', 'e2', 'i2', 'f2', 'g6', 'e5', 'e3', 'i3', 'h3']

# 0626_005617: Groucho vs. Chico, groucho forms a triangle, and forces it... but forces chico to win.
game_so_far = ['d5', 'e6', 'i4', 'b3', 'e5', 'd3', 'f3', 'b2', 'b6', 'h2', 'i1', 'e3', 'c3', 'i3', 'f7', 'g5', 'c1', 'c7', 'f6', 'h1', 'b1', 'c4', 'h4', 'f1', 'h5', 'g6', 'd8', 'e1', 'g1', 'e4', 'e2', 'h6', 'a5', 'i5', 'd6', 'd7', 'b5', 'c5', 'c6']
