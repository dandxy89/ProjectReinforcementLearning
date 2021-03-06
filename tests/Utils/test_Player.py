#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Testing for Utils
"""
import datetime
import unittest

from RLBook.Utils.Player import Player


class TestUtilsPlayer(unittest.TestCase):
    """ Testing the Utils Implementations
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_player_init(self):
        """ TODO
        """
        p1 = Player(name="Dan", value=89, display="D")

        assert isinstance(p1, Player)
        assert p1.__str__() == "< Player Dan | Label 89 | Display D >"
        assert p1.__repr__() == "< Player Dan | Label 89 | Display D >"

    def test_player_equality(self):
        """ TODO
        """
        p2 = Player(name="Dan", value=89, display="D")
        p3 = Player(name="Dan", value=89, display="D")
        p4 = Player(name="Dan", value=90, display="D")

        assert p2 == p3
        assert p2 != p4

    def test_player_property(self):
        """ TODO
        """
        p5 = Player(name="Dan", value=90, display="D")

        assert p5.mcts_search == (10000, 8)
        print(p5.nn_params)
        assert p5.nn_params == {'MCTS_MAX_TIME': 8, 'ACTIVATION_DENSE': 'tanh', 'CNN_FILTER_SIZE': 1,
                                'MODEL_TYPE': 'ResNet',
                                'VALUE_FC_SIZE': 1, 'ACTIVATION': 'relu',
                                'MODEL_NAME': datetime.datetime.now().strftime("%Y%m%d_KerasModel_TTT_V"),
                                'RES_LAYER_NUM': 0,
                                'CNN_FILTER_NUM': 2, 'EPOCHS': 3, 'ACTIVATION_POLICY': 'softmax', 'L2_REG': 0.0002,
                                'MCTS_ITERATIONS': 10000,
                                'BATCH_SIZE': 8, 'N_LABELS': 9}

        assert not p5.use_nn


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner())
