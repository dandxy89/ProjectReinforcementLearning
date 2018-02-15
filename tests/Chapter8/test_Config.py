#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Testing for Chapter 8 Config
"""
import datetime
import unittest

from RLBook.Chapter8.Config import Config, EnvConfig


class TestChapter8Config(unittest.TestCase):
    """ Testing the Chapter8 Implementations
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_initialise_agent(self):
        """ Test the Initialisation of the Agent Config
        """
        c = Config(**dict(MODEL_NAME="TESTING"))

        assert c.__repr__() == '< Player Config >'
        assert isinstance(c, Config)
        assert c.MODEL_NAME == "TESTING"

    def test_agent_mcts_params(self):
        """ TODO
        """
        c1 = Config()

        assert c1.mcts_params == {'Q': 0.0, 'TAU': 1.0, 'C_PUCT': 2.0, 'N_PLAYS': 0, 'U': 0.0, 'ACTION': None,
                                  'N_WINS': 0, 'V': 0.0, 'SCORE': 0.0, 'N_TIES': 0, 'PRIOR': 1.0}

    def test_agent_mcts_search(self):
        """ TODO
        """
        c2 = Config()

        assert c2.mcts_search == (10000, 8)

    def test_agent_nn_params(self):
        """ TODO
        """
        c3 = Config()

        assert c3.nn_params == {'N_LABELS': 9, 'BATCH_SIZE': 8, 'CNN_FILTER_SIZE': 1, 'MCTS_MAX_TIME': 8, 'EPOCHS': 3,
                                'RES_LAYER_NUM': 0,
                                'L2_REG': 0.0002, 'MCTS_ITERATIONS': 10000, 'ACTIVATION': 'relu',
                                'MODEL_NAME': '{}_KerasModel_TTT_V'.format(datetime.datetime.now().strftime("%Y%m%d")),
                                'ACTIVATION_DENSE': 'tanh', 'VALUE_FC_SIZE': 1, 'CNN_FILTER_NUM': 2,
                                'MODEL_TYPE': 'ResNet',
                                'ACTIVATION_POLICY': 'softmax'}

    def test_agent_get_all(self):
        """ TODO
        """
        c4 = Config()

        assert c4.get_all == {'N_PLAYS': 0, 'ACTIVATION': 'relu', 'N_WINS': 0, 'VALUE_FC_SIZE': 1,
                              'MODEL_TYPE': 'ResNet',
                              'ACTIVATION_DENSE': 'tanh', 'PRIOR': 1.0, 'N_LABELS': 9,
                              'MODEL_NAME': '{}_KerasModel_TTT_V'.format(datetime.datetime.now().strftime("%Y%m%d")),
                              'BATCH_SIZE': 8, 'MCTS_ITERATIONS': 10000, 'V': 0.0, 'CNN_FILTER_NUM': 2,
                              'MCTS_MAX_TIME': 8, 'Q': 0.0,
                              'RES_LAYER_NUM': 0, 'U': 0.0, 'L2_REG': 0.0002, 'N_TIES': 0, 'ACTION': None,
                              'CNN_FILTER_SIZE': 1,
                              'ACTIVATION_POLICY': 'softmax', 'EPOCHS': 3, 'TAU': 1.0, 'C_PUCT': 2.0, 'SCORE': 0.0}

    def test_initialise_env(self):
        """ Test the Initiation of the Env class
        """
        e = EnvConfig(**dict(N_ITERATION=100))

        assert e.__repr__() == '< Environment Config >'
        assert isinstance(e, EnvConfig)
        assert e.N_ITERATION == 100

        e1 = EnvConfig()

        assert e1.__repr__() == '< Environment Config >'
        assert isinstance(e1, EnvConfig)
        assert e1.N_ITERATION == 30

    def test_env_get_all(self):
        """ TODO
        """
        e2 = EnvConfig()

        # Drop Start T and Start Dt
        d = e2.get_all
        del d["START_T"]
        del d["START_TIME"]

        assert d == {'N_EPISODE': 40, 'WIN_RATIO': 0.3, 'EVALUATIONS': 20, 'N_ITERATION': 30}


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner())
