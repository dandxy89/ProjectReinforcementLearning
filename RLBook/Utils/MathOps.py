# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.Utils.MathOps
"""
import keras.backend as K
import numpy as np
from keras.losses import mean_squared_error
from scipy.stats import beta


def softmax(x: np.ndarray) -> np.ndarray:
    """ Softmax function

        :param x:       Input Array
        :return:        Array with the Softmax function applied

    """
    x1 = np.exp(x - np.max(x))
    x1 /= np.sum(x1)

    return x1


def thompson(total_plays, node):
    """ Thompson sampling

        :param total_plays: number of plays of all arms
        :param node:        node properties
        :return:            score (min:0, max:1)

    """
    # Set the value high to ensure this edge is select
    if node.N_PLAYS == 0:
        return 99.
    else:
        return beta.rvs(a=node.WINS + 1, b=node.N_PLAYS - node.N_WINS + 1, size=1)[0]


def upper_confidence_bound(total_plays, node):
    """ Upper Confidence Bound used by AlphaZero

        :param total_plays: number of plays of all arms
        :param node:        node properties
        :return:            score (min:0, max:1)

    """
    if node.Q == 0:
        node.U = node.C_PUCT * node.PRIOR * np.sqrt(node.N_PLAYS)
    else:
        node.U = node.Q + node.C_PUCT * node.PRIOR * np.sqrt(node.parent.N_PLAYS) / (1 + node.N_PLAYS)

    return node.U


def random_value_policy(state):
    return [(val, prob / 2) for val, prob in enumerate(np.ones(9))], 1.


def objective_function_for_policy(y_true, y_pred):
    return K.sum(-y_true * K.log(y_pred + K.epsilon()), axis=-1)


def objective_function_for_value(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
