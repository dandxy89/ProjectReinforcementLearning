#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Testing for Utils
"""
import unittest

import numpy as np
from anytree import Node

from RLBook.Chapter8 import DEFAULT_NODE_PARAMS
from RLBook.Utils.MathOperations import softmax, upper_confidence_bound


class TestMathOps(unittest.TestCase):
    """ Testing the Utils Implementations
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_softmax(self):
        """ TODO
        """
        a = np.ones(9)

        assert softmax(a).sum() == 1

    def test_alpha_zero_ucb(self):
        """
        """
        node1 = Node('A', GAME=1, **DEFAULT_NODE_PARAMS)
        node1.N_PLAYS = 1

        assert upper_confidence_bound(node=node1) == 2.

        node2 = Node('A', GAME=1, **DEFAULT_NODE_PARAMS)
        node2.Q = 1.2
        node2.N_PLAYS = 20
        node2.parent = node1

        assert upper_confidence_bound(node=node2) == 1.295238095238095


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner())
