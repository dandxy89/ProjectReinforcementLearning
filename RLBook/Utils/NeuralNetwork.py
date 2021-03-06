#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.Utils

- NeuralNetwork Base Class definition

"""
import logging
from abc import abstractclassmethod, ABCMeta

from keras import backend as K
from keras.losses import mean_squared_error
from keras.models import model_from_json


class NeuralNet:
    """ This class specifies the base NeuralNet class.
    """
    __metaclass__ = ABCMeta

    def __init__(self, check_point=0):
        """ Initialise a NeuralNetwork

            :param check_point:         Counter that will be used when save the NN

        """
        self.CHECK_POINT = check_point
        self.model = None

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

    def save_checkpoint(self, filename: str):
        """ Saves the current neural network (with its parameters) into a given filename
        """
        # serialize model to JSON
        model_json = self.model.to_json()
        with open('{}.json'.format(filename), "w") as json_file:
            json_file.write(model_json)

        # Serialize weights to HDF5
        self.model.save_weights("{}.h5".format(filename))
        logging.info("Model has been check-pointed: {}".format(filename))

    def load_checkpoint(self, filename):
        """ Loads parameters of the neural network from a given filename
        """
        with open('{}.json'.format(filename), 'r') as json_file:
            loaded_model_json = json_file.read()

        self.model = model_from_json(loaded_model_json)

        # load weights into new model
        self.model.load_weights("{}.h5".format(filename))
        logging.info("Model has been loaded from a check-pointed: {}".format(filename))

    @property
    def increment(self):
        return self.CHECK_POINT

    @increment.setter
    def increment(self, value):
        self.CHECK_POINT = value


def objective_function_for_policy(y_true, y_pred):
    return K.sum(-y_true * K.log(y_pred + K.epsilon()), axis=-1)


def objective_function_for_value(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
