#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Testing for Chapter 3
"""
import unittest

from RLBook.Chapter3.Gridworld import GridWorld


class TestGridWorld(unittest.TestCase):
    """ Testing the Chapter3 Implementation
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_initialisation(self):
        """ Testing the Initialisation of the GridWorld class
        """
        # Initialise the Grid World
        env = GridWorld(**{"TOL": 1e-8})

        env.find_possible_states()

        assert len(env.NEXT_STATE) == 5
        assert len(env.ACTION_REWARD) == 5


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner())
