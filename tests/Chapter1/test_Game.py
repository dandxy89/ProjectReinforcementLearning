#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Testing for Chapter 1
"""
import unittest

from RLBook.Chapter1.TicTacToe import TicTacToeGame


class TestTicTacToe(unittest.TestCase):
    """ Testing the Chapter1 Implementation
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_initialise(self):
        game_one = TicTacToeGame()

        # Assertions
        self.assertEquals(game_one.BOARD.sum(), 0)
        self.assertEquals(game_one.WINNER, None)
        self.assertEquals(game_one.BOARD_COLS, 3)
        self.assertEquals(game_one.BOARD_ROWS, 3)

    def test_getting_state(self):
        game_two = TicTacToeGame()
        game_two.BOARD[0, 0] = 1

        assert game_two.get_state() == '[1, 0, 0, 0, 0, 0, 0, 0, 0]'

    def test_check_row(self):
        game_three = TicTacToeGame()
        game_three.BOARD[0, 0] = 1

        # Should return False
        assert game_three.get_state() == '[1, 0, 0, 0, 0, 0, 0, 0, 0]'
        assert not game_three.is_end_state()

        game_three.BOARD[0, 1] = 1
        game_three.BOARD[0, 2] = 1

        # Assertions
        assert game_three.is_end_state()
        assert game_three.WINNER == 1

    def test_check_column(self):
        game_four = TicTacToeGame()
        game_four.BOARD[0, 0] = 1

        # Should return False
        assert not game_four.is_end_state()

        game_four.BOARD[1, 0] = 1
        game_four.BOARD[2, 0] = 1

        # Assertions
        assert game_four.is_end_state()
        assert game_four.WINNER == 1

    def test_check_diagonal(self):
        game_five = TicTacToeGame()
        game_five.BOARD[0, 0] = 1

        # Should return False
        assert not game_five.is_end_state()

        game_five.BOARD[1, 1] = 1
        game_five.BOARD[2, 2] = 1

        # Assertions
        assert game_five.is_end_state()
        assert game_five.WINNER == 1

    def test_check_opposite_diagonal(self):
        game_five = TicTacToeGame()
        game_five.BOARD[0, 2] = 1

        # Should return False
        assert not game_five.is_end_state()

        game_five.BOARD[1, 1] = 1
        game_five.BOARD[2, 0] = 1

        # Assertions
        assert game_five.is_end_state()
        assert game_five.WINNER == 1

    def test_check_draw(self):
        game_six = TicTacToeGame()
        game_six.BOARD[0, 0] = 1
        game_six.BOARD[1, 0] = 1
        game_six.BOARD[2, 0] = -1
        game_six.BOARD[0, 1] = -1
        game_six.BOARD[1, 1] = -1
        game_six.BOARD[2, 1] = 1
        game_six.BOARD[0, 2] = 1
        game_six.BOARD[1, 2] = -1
        game_six.BOARD[2, 2] = -1

        # Assertions
        assert game_six.is_end_state()
        assert game_six.WINNER == 0


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner())
