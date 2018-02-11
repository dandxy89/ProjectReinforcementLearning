# -*- coding: utf-8 -*-
""" RLBook.Chapter8.TicTacToe

Tic Tac Toe game implementation

"""
from itertools import cycle

import numpy as np

from RLBook.Chapter8.DefaultPlayers import DEFAULT_PLAYERS


class Game:
    """ TicTacToe game implementation to be used by Monte Carlo Tree Search

        https://en.wikipedia.org/wiki/Tic-tac-toe

    """
    DICTIONARY = {(0, 0): 0,
                  (0, 1): 1,
                  (0, 2): 2,
                  (1, 0): 3,
                  (1, 1): 4,
                  (1, 2): 5,
                  (2, 0): 6,
                  (2, 1): 7,
                  (2, 2): 8}
    INDEX = {1: 0,
             0: 1}

    players = DEFAULT_PLAYERS

    def __init__(self, board_size=3, players=None, using_nn=None, nn_player=0):
        """

            :param board_size:          3x3 by Default will be the Board size used
            :param players:             Optional: Pass
            :param using_nn:            Flag to indicate if a Neural Network is being utilised
            :param nn_player:           Identification of which agent is the Neural Network

        """
        # Game attributes
        self.board_size = board_size
        self.state = np.zeros((board_size, board_size), dtype=int)
        self.last_play = None
        self.sums = np.array([])

        # players attributes
        if players is not None:
            self.players = players

        self.players_values = list([p.value for p in self.players])
        self.players_gen = cycle(self.players)
        self.current_player = next(self.players_gen)
        self.history = [(self.state.copy(), None, self.current_player.value, None)]

        # Using a Neural Network
        self.nn_player = nn_player
        self.using_nn = using_nn

    def __repr__(self):
        return "< TicTacToe > "

    def __str__(self):
        return "< TicTacToe > "

    def legal_plays(self):
        """ Takes a sequence of game states representing the full game history

            :return:        the list of moves tuples that are legal to play for the current player

        """
        legal_plays = []
        if self.winner is None:
            free_spaces = np.isin(self.state, self.players_values, invert=True)
            legal_plays = np.argwhere(free_spaces)

            # convert numpy array to list of tuples
            legal_plays = list(map(tuple, legal_plays))

        return legal_plays

    @property
    def winner(self):
        """ Return the winner player. If game is tied, return None

            :return:        Player or None

        """
        for player in self.players:
            # one axis is full of this player plays (= win)
            if self.board_size * player.value in self.sums:
                return player

        # no winner found
        return None

    def show_board(self, state_number=-1, return_string=False):
        """ Display the game board

            :param state_number:        the state to show
            :param return_string:       whether to return a string or to print it
            :return:                    board representation as a string or nothing

        """
        # creates the string representation of the game
        lines = []
        no_player_display = '.'
        for line in self.history[state_number][0]:
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

    def play(self, move=None, action_prob=1):
        """ Play a move

            :param move:            selected move to play. If None it is chosen randomly from legal plays
            :param action_prob:     Action probabilities from the Agent

        """
        legal_plays = self.legal_plays()

        # If input move is provided check that it is legal
        if move is not None:
            if move in legal_plays:
                selected_move = move
            else:
                raise ValueError('Selected move is illegal')
        # Select a move randomly
        else:
            selected_move = legal_plays[np.random.choice(len(legal_plays), 1)[0]]

        # Updates states and players info
        self.state[selected_move] = self.current_player.value

        # Copy() needed to avoid appending a reference
        # noinspection PyTypeChecker
        self.history.append((self.state.copy(), self.translate(selected_move), self.current_player.value, action_prob))
        self.current_player = next(self.players_gen)
        self.last_play = selected_move

        # Updates sums that are used to check for winner
        self.sums = np.concatenate(
            (np.sum(self.state, axis=0),  # vertical
             np.sum(self.state, axis=1),  # horizontal
             np.array([np.sum(np.diag(self.state)),  # diagonal
                       np.sum(np.diag(self.state[::-1]))])))

    def translate(self, position):
        """ Translate tuple to Index

            :param position:
            :return:

        """
        return self.DICTIONARY.get(position)

    def reset(self):
        # Game attributes
        self.state = np.zeros((self.board_size, self.board_size), dtype=int)
        self.history = [self.state.copy()]  # copy() needed to avoid appending a reference
        self.last_play = None
        self.sums = np.array([])

    @property
    def nn_index(self):
        """ Get the Neural Network Players Coin choice and index in the Player

            :return:        NN Player, Index in List of Players, Player

        """
        if self.using_nn:
            return None
        else:
            return self.players[self.nn_player], self.nn_player

    @property
    def player(self):
        return self.current_player

    @property
    def competing_player(self):
        return self.players[self.INDEX[self.nn_player]]
