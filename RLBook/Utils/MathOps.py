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
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)

    return probs
