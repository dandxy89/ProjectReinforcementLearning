# -*- coding: utf-8 -*-
""" RLBook.Chapter8.TicTacToe

Tic Tac Toe game implementation

"""
import logging
from itertools import cycle

import numpy as np


class Player:
    """ TicTacToe player
    """

    def __init__(self, name, value, display):
        self.name = name
        self.value = value
        self.display = display

    def __eq__(self, other):
        """ Two players are equal if they have the same value
        """
        if isinstance(self, other.__class__):
            return self.value == other.value

        return False


class Game:
    """ TicTacToe game implementation to be used by Monte Carlo Tree Search

        https://en.wikipedia.org/wiki/Tic-tac-toe

    """
    dictionary = {(0, 0): 0,
                  (0, 1): 1,
                  (0, 2): 2,
                  (1, 0): 3,
                  (1, 1): 4,
                  (1, 2): 5,
                  (2, 0): 6,
                  (2, 1): 7,
                  (2, 2): 8}

    def __init__(self, board_size=3, save_history=True):
        """

            :param board_size:
            :param save_history:

        """
        # Game attributes
        self.board_size = board_size
        self.state = np.zeros((board_size, board_size), dtype=int)
        self.save_history = save_history
        self.history = [self.state.copy()]  # copy() needed to avoid appending a reference
        self.last_play = None
        self.sums = np.array([])

        # players attributes
        self.players = [Player(name='A', value=1, display='O'), Player(name='B', value=-1, display='X')]
        self.players_values = list([p.value for p in self.players])
        self.players_gen = cycle(self.players)
        self.current_player = next(self.players_gen)

    def __repr__(self):
        return "< TicTacToe > "

    def __str__(self):
        return "< TicTacToe > "

    def legal_plays(self):
        """ Takes a sequence of game states representing the full game history

            :return:        the list of moves tuples that are legal to play for the current player

        """
        legal_plays = []
        if self.winner() is None:
            free_spaces = np.isin(self.state, self.players_values, invert=True)
            legal_plays = np.argwhere(free_spaces)

            # convert numpy array to list of tuples
            legal_plays = list(map(tuple, legal_plays))

        logging.debug('Legal plays: %s', legal_plays)
        return legal_plays

    def winner(self):
        """ Return the winner player. If game is tied, return None

            :return:        Player or None

        """
        for player in self.players:
            # one axis is full of this player plays (= win)
            if self.board_size * player.value in self.sums:
                logging.debug('Winner: %s', player.display)
                return player

        logging.debug('Winner: None')
        return None  # no winner found

    def show_board(self, state_number=-1, return_string=False):
        """ Display the game board

            :param state_number:        the state to show
            :param return_string:       whether to return a string or to print it
            :return:                    board representation as a string or nothing

        """
        # creates the string representation of the game
        lines = []
        no_player_display = '.'
        for line in self.history[state_number]:
            elements = []
            for element in line:
                if element in self.players_values:
                    for player in self.players:
                        if element == player.value:
                            elements.append(player.display)
                else:
                    elements.append(no_player_display)
            lines.append('|'.join(elements))
        board_representation = '\n'.join(lines)

        if return_string:
            return board_representation
        else:
            print(board_representation)

    def play(self, move=None):
        """ Play a move

            :param move: selected move to play. If None it is chosen randomly from legal plays

        """
        legal_plays = self.legal_plays()

        # if input move is provided check that it is legal
        if move is not None:
            if move in legal_plays:
                selected_move = move
            else:
                raise ValueError('Selected move is illegal')
        # select a move randomly
        else:
            selected_move = legal_plays[np.random.choice(len(legal_plays), 1)[0]]

        # Updates states and players info
        self.state[selected_move] = self.current_player.value
        if self.save_history:
            # Copy() needed to avoid appending a reference
            self.history.append(self.state.copy())

        # Only the current state is save (to be able to display it)
        else:
            self.history = [self.state.copy()]

        self.current_player = next(self.players_gen)
        self.last_play = selected_move

        # Updates sums that are used to check for winner
        self.sums = np.concatenate(
            (np.sum(self.state, axis=0),  # vertical
             np.sum(self.state, axis=1),  # horizontal
             np.array([np.sum(np.diag(self.state)),  # diagonal
                       np.sum(np.diag(self.state[::-1]))])))

    def translate(self, position):
        return self.dictionary.get(position)
