#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Testing for Utils
"""
import unittest

from RLBook.Utils.Trainer import AllConfig, Trainer


class TestUtilsTrainer(unittest.TestCase):
    """ Testing the Utils Implementations
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_all_config(self):
        """ TODO
        """
        a = AllConfig()

        assert a.__repr__() == "< Env / Game & Trainer Configuration >"

        # Test setter
        a.trainer = "DansProject"

        assert a.trainer == "DansProject"
        assert isinstance(a, AllConfig)

    def test_trainer_init(self):
        """ Trainer class init
        """
        t = Trainer(environment="Game", trainer_config="MSc Dan", agents=["A", "B"])

        assert isinstance(t, Trainer)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner())
