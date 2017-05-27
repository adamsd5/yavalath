"""Unit tests for yavalath.py"""
import unittest
import yavalath

class TestYavalath(unittest.TestCase):
    def test_connected_along_axis(self):
        board = yavalath.HexBoard()
        self.assertEqual(yavalath._connected_along_axis(board, "e6", ["e4", "a1", "e5", "a2"], 0), 3)
        self.assertEqual(3, yavalath._connected_along_axis(board, "g1", ["e1", "f1"], 1))
        self.assertEqual(3, yavalath._connected_along_axis(board, "g7", ['e7', 'c1', 'f7', 'e9', 'e6', 'h6'], 1))
        self.assertEqual(3, yavalath._connected_along_axis(board, "c3", ['f8', 'd8', 'b6', 'f6', 'c1', 'd2', 'f3', 'g4', 'c2', 'd5', 'f4', 'e7', 'g6', 'e8', 'i5', 'h1', 'h4', 'd4', 'd6', 'a4', 'e4', 'h6'], 0))

    def test_judge_next_move(self):
        self.assertEqual(yavalath.MoveResult.PLAYER_LOSES, yavalath.judge_next_move(['f8', 'd8', 'b6', 'f6', 'c1', 'd2', 'f3', 'g4', 'c2', 'd5', 'f4', 'e7', 'g6', 'e8', 'i5', 'h1', 'h4', 'd4', 'd6', 'a4', 'e4', 'h6'], "c3"))
        self.assertEqual(yavalath.MoveResult.PLAYER_LOSES, yavalath.judge_next_move(['c1', 'g5', 'a1', 'a4', 'e5', 'i1', 'h4', 'g3', 'd3', 'e3', 'b2', 'e2'], "c3"))

        #['f8', 'd8', 'b6', 'f6', 'c1', 'd2', 'f3', 'g4', 'c2', 'd5', 'f4', 'e7', 'g6', 'e8', 'i5', 'h1', 'h4', 'd4', 'd6', 'a4', 'e4', 'h6', 'c3']

class TestHexBoard(unittest.TestCase):
    def test_next_space(self):
        b = yavalath.HexBoard()
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