"""Unit tests for yavalath.py"""
import unittest
import yavalath_engine

class TestYavalath(unittest.TestCase):
    def test_connected_along_axis(self):
        board = yavalath_engine.HexBoard()
        self.assertEqual(yavalath_engine._connected_along_axis(board, "e6", ["e4", "a1", "e5", "a2"], 0), 3)
        self.assertEqual(3, yavalath_engine._connected_along_axis(board, "g1", ["e1", "f1"], 1))
        self.assertEqual(3, yavalath_engine._connected_along_axis(board, "g7", ['e7', 'c1', 'f7', 'e9', 'e6', 'h6'], 1))
        self.assertEqual(3, yavalath_engine._connected_along_axis(board, "c3", ['f8', 'd8', 'b6', 'f6', 'c1', 'd2', 'f3', 'g4', 'c2', 'd5', 'f4', 'e7', 'g6', 'e8', 'i5', 'h1', 'h4', 'd4', 'd6', 'a4', 'e4', 'h6'], 0))

    def test_judge_next_move(self):
        self.assertEqual(yavalath_engine.MoveResult.PLAYER_LOSES, yavalath_engine.judge_next_move(['f8', 'd8', 'b6', 'f6', 'c1', 'd2', 'f3', 'g4', 'c2', 'd5', 'f4', 'e7', 'g6', 'e8', 'i5', 'h1', 'h4', 'd4', 'd6', 'a4', 'e4', 'h6'], "c3"))
        self.assertEqual(yavalath_engine.MoveResult.PLAYER_LOSES, yavalath_engine.judge_next_move(['c1', 'g5', 'a1', 'a4', 'e5', 'i1', 'h4', 'g3', 'd3', 'e3', 'b2', 'e2'], "c3"))

        #['f8', 'd8', 'b6', 'f6', 'c1', 'd2', 'f3', 'g4', 'c2', 'd5', 'f4', 'e7', 'g6', 'e8', 'i5', 'h1', 'h4', 'd4', 'd6', 'a4', 'e4', 'h6', 'c3']

    def test_renderer_debug(self):
        move_stack = ['d2', 'b2', 'd4', 'f5', 'd5', 'd3', 'b5', 'c5', 'f1', 'e2', 'f2', 'a1', 'b1', 'c7', 'c6', 'e8', 'g6', 'f6', 'b6', 'd6', 'a3', 'g1']
        game_so_far = ['g2', 'c2', 'g7', 'i5', 'd1', 'e6', 'c4', 'a2', 'b3', 'e5', 'g4', 'g5', 'c1'] + move_stack
        yavalath_engine.Render(board=yavalath_engine.HexBoard(), moves=game_so_far).render_image("debug.png")


class TestHexBoard(unittest.TestCase):
    def test_next_space(self):
        b = yavalath_engine.HexBoard()
        self.assertEqual("e6", b.next_space("e5", 0, 1))
        self.assertEqual("e4", b.next_space("e5", 0, -1))
        self.assertEqual("f5", b.next_space("e5", 1, 1))
        self.assertEqual("d4", b.next_space("e5", 1, -1))
        self.assertEqual("d5", b.next_space("e5", -1, 1))
        self.assertEqual("f4", b.next_space("e5", -1, -1))

        self.assertEqual("d6", b.next_space("d5", 0, 1))
        self.assertEqual("d4", b.next_space("d5", 0, -1))
        self.assertEqual("e6", b.next_space("d5", 1, 1))
        self.assertEqual("c4", b.next_space("d5", 1, -1))
        self.assertEqual("c5", b.next_space("d5", -1, 1))
        self.assertEqual("e5", b.next_space("d5", -1, -1))

        self.assertEqual("f6", b.next_space("f5", 0, 1))
        self.assertEqual("f4", b.next_space("f5", 0, -1))
        self.assertEqual("g5", b.next_space("f5", 1, 1))
        self.assertEqual("e5", b.next_space("f5", 1, -1))
        self.assertEqual("e6", b.next_space("f5", -1, 1))
        self.assertEqual("g4", b.next_space("f5", -1, -1))

        self.assertEqual("f1", b.next_space("g1", 1, -1))
