# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.Utils.MathOps
"""
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """ Softmax function

        :param x:       Input Array
        :return:        Array with the Softmax function applied

    """
    x1 = np.exp(x - np.max(x))
    x1 /= np.sum(x1)

    return x1


def upper_confidence_bound(node):
    """ Upper Confidence Bound used by AlphaZero

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
