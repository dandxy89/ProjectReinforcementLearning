# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" TicTacToe.TicTacToe
"""
import numpy as np

BOARD_ROWS = 3
BOARD_COLS = 3


class GameState:
    """ TicTacToe Implementation
    """
    BOARD_ROWS = BOARD_ROWS
    BOARD_COLS = BOARD_COLS
    WINNER = None
    BOARD = None

    def __init__(self, rows=BOARD_ROWS, columns=BOARD_COLS):
        """ Initialise the Game of TicTacToe.

            The Game can be represented by a cols * rows 2D array. For this model one player (starting) will be
            represented by 1 and the other player by -1, finally, 0 will represented unselected space.

            :param rows:
            :param columns:

        """
        # Setup the Board
        self.BOARD_ROWS = rows
        self.BOARD_COLS = columns
        self.BOARD = np.zeros((self.BOARD_ROWS, self.BOARD_COLS))

    def __repr__(self):
        return "< TicTacToe [{}, {}] >".format(self.BOARD_ROWS, self.BOARD_COLS)

    @staticmethod
    def check_line(array, axis=None):
        """ Check both Rows and Columns

            :param array:
            :param axis:
            :return:

        """
        if np.any(np.sum(array, axis=axis) == 3):
            return True, 1
        elif np.any(np.sum(array, axis=axis) == -3):
            return True, -1
        else:
            return False, None

    @staticmethod
    def check_opposite_diagonal(array):
        """ Check the Upper Left Diagonal

            :param array:
            :return:

        """
        if np.sum(np.diag(np.fliplr(array))) == 3:
            return True, 1
        elif np.sum(np.diag(np.fliplr(array))) == -3:
            return True, -1
        else:
            return False, None

    @staticmethod
    def check_diagonal(array):
        """ Check the Upper Right Diagonal

            :param array:
            :return:

        """
        if np.sum(np.diag(array)) == 3:
            return True, 1
        elif np.sum(np.diag(array)) == -3:
            return True, -1
        else:
            return False, None

    def get_state(self):
        """ Get the current state of the Board
        """
        return self.BOARD.flatten()

    def display(self):
        print(self.BOARD)

    def is_end_state(self):
        """ Determine whether a player has won the game, or it's a tie
        """
        # Check Rows
        win, winner = self.check_line(self.BOARD, axis=1)

        # Check Columns
        win2, winner2 = self.check_line(self.BOARD, axis=0)

        # Check Diagonal
        win3, winner3 = self.check_diagonal(self.BOARD)

        # Check Opposite Diagonal
        win4, winner4 = self.check_opposite_diagonal(self.BOARD)

        # Check if Player A or Player B is the Winner
        if np.any([win, win2, win3, win4]):
            a = np.array([winner, winner2, winner3, winner4])
            self.WINNER = a[a.nonzero()][0]
            return True
        # Check for the Draw...
        elif np.abs(self.BOARD).sum() == (self.BOARD_COLS * self.BOARD_ROWS):
            self.WINNER = 0
            return True
        # Other State where there is neither a Winner or a Draw
        else:
            return False
