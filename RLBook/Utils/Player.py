# -*- coding: utf-8 -*-
""" RLBook.Utils.Player
"""
from RLBook.Chapter8.Config import Config
from RLBook.Utils.MathOps import random_value_policy


class Player:
    """ A Game Player
    """
    eval_function = random_value_policy

    def __init__(self, name, value, display, fn=None, use_nn=False, config=Config):
        self.name = name
        self.value = value
        self.display = display

        if use_nn:
            from RLBook.Chapter8.KerasModel import KerasModel
            self.nn = use_nn
            self.eval_function = KerasModel(config.nn_params).predict
        else:
            self.nn = False

        # Get all the Config
        self.c = config()

        if fn is not None:
            self.eval_function = fn

    def __eq__(self, other):
        """ Two players are equal if they have the same value
        """
        if isinstance(self, other.__class__):
            return self.value == other.value
        return False

    def __repr__(self):
        return "< Player {} | Coin {} | Display {} >".format(self.name, self.value, self.display)

    def __str__(self):
        return "< Player {} | Coin {} | Display {} >".format(self.name, self.value, self.display)

    @property
    def func(self):
        return self.eval_function

    @property
    def use_nn(self):
        return self.nn

    @property
    def mcts_search(self):
        return self.c.MCTS_ITERATIONS, self.c.MCTS_MAX_TIME

    @property
    def mcts_params(self):
        return self.c.mcts_params

    @property
    def nn_params(self):
        return self.c.nn_params
