#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Testing for Utils
"""
import unittest

from RLBook.Utils.Trainer import Trainer


class TestUtilsTrainer(unittest.TestCase):
    """ Testing the Utils Implementations
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_trainer_init(self):
        """ Trainer class init
        """
        t = Trainer(environment="Game", trainer_config="MSc Dan")

        assert isinstance(t, Trainer)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner())
