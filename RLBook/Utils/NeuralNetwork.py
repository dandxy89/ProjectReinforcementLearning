#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.Utils

- NeuralNetwork Base Class definition

"""
from abc import abstractclassmethod, ABCMeta


class NeuralNet:
    """ This class specifies the base NeuralNet class.
    """
    __metaclass__ = ABCMeta

    def __init__(self, game):
        pass

    @abstractclassmethod
    def train(self, examples):
        """ This function trains the neural network with examples obtained from self-play.

            @:param  examples:  a list of training examples, where each example is of form
                                (board, pi, v). pi is the MCTS informed policy vector for
                                the given board, and v is its value. The examples has
                                board in its canonical form.
        """
        pass

    @abstractclassmethod
    def predict(self, board):
        """ Predict given a board state

            :param board:   Current board in its canonical form.
            :returns: pi:   A list of (action, pi) tuples
                            v: a float in [-1,1] that gives the value of the current board
        """
        pass

    @abstractclassmethod
    def save_checkpoint(self, filename):
        """ Saves the current neural network (with its parameters) into a given filename
        """
        pass

    @abstractclassmethod
    def load_checkpoint(self, filename):
        """ Loads parameters of the neural network from a given filename
        """
        pass
