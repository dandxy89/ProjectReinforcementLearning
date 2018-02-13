# -*- coding: utf-8 -*-
""" Chapter8.KerasModel

-   Keras Model - Replicating the work by AlphaZero

"""
import numpy as np

from RLBook.Utils.NeuralNetwork import NeuralNet


class RandomModel(NeuralNet):
    """ Keras Model definition
    """

    def __init__(self, config=None):
        """ Initialise a Random Playing Model

            Note: Config is not used...

            :param config:

        """
        super().__init__()
        self.config = config

    def __repr__(self):
        return "< Random Model {} >".format(self.config.MODEL_NAME)

    def __str__(self):
        return "< Random Model {} >".format(self.config.MODEL_NAME)

    def train(self, tuple_arrays):
        """ Method to replicate the Keras Model version

            :param tuple_arrays:
            :return:

        """
        pass

    def predict(self, tuple_arrays=None, state=None, current_player=0):
        """ Prediction

            :param tuple_arrays:    Handling a large batch
            :param state:           Handling a single state instance directly from the game
            :param current_player:  Current Player in index position
            :return:

        """
        return [(val, prob) for val, prob in enumerate(np.random.rand(9))], np.random.random()
