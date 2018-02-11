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

        assert c1.mcts_params == {'Q': 0.0, 'PRIOR': 1.0, 'SCORE': 0.0, 'N_WINS': 0, 'ACTION': None, 'C_PUCT': 2.0,
                                  'N_PLAYS': 0, 'TAU': 1.0, 'U': 0.0, 'N_TIES': 0}

    def test_agent_mcts_search(self):
        """ TODO
        """
        c2 = Config()

        assert c2.mcts_search == (30000, 20)

    def test_agent_nn_params(self):
        """ TODO
        """
        c3 = Config()

        assert c3.nn_params == {'VALUE_FC_SIZE': 1, 'EPOCHS': 2, 'ACTIVATION_POLICY': 'softmax', 'MODEL_TYPE': 'ResNet',
                                'N_LABELS': 9, 'CNN_FILTER_NUM': 2, 'L2_REG': 0.0002, 'CNN_FILTER_SIZE': 1,
                                'MODEL_NAME': '{}_KerasModel_TTT'.format(datetime.datetime.now().strftime("%Y%m%d")),
                                'MCTS_MAX_TIME': 20, 'ACTIVATION': 'relu',
                                'RES_LAYER_NUM': 0, 'BATCH_SIZE': 1, 'MCTS_ITERATIONS': 30000,
                                'ACTIVATION_DENSE': 'tanh'}

    def test_agent_get_all(self):
        """ TODO
        """
        c4 = Config()

        assert c4.get_all == {'CNN_FILTER_NUM': 2, 'ACTIVATION': 'relu', 'N_PLAYS': 0,
                              'MODEL_NAME': '{}_KerasModel_TTT'.format(datetime.datetime.now().strftime("%Y%m%d")),
                              'ACTION': None, 'MCTS_MAX_TIME': 20,
                              'RES_LAYER_NUM': 0, 'TAU': 1.0, 'PRIOR': 1.0, 'N_TIES': 0, 'ACTIVATION_DENSE': 'tanh',
                              'ACTIVATION_POLICY': 'softmax', 'N_LABELS': 9, 'U': 0.0, 'C_PUCT': 2.0,
                              'MCTS_ITERATIONS': 30000, 'MODEL_TYPE': 'ResNet', 'BATCH_SIZE': 1, 'Q': 0.0,
                              'CNN_FILTER_SIZE': 1, 'L2_REG': 0.0002, 'VALUE_FC_SIZE': 1, 'EPOCHS': 2, 'N_WINS': 0,
                              'SCORE': 0.0}

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
        assert e1.N_ITERATION == 5

    def test_env_get_all(self):
        """ TODO
        """
        e2 = EnvConfig()

        # Drop Start T and Start Dt
        d = e2.get_all
        del d["START_T"]
        del d["START_TIME"]

        assert d == {'CHECKPOINT': 0, 'N_ITERATION': 5, 'N_EPISODE': 5, 'EVALUATIONS': 10, 'WIN_RATIO': 0.55}

    def test_check_property(self):
        """ Checkpoint Attribute
        """
        e3 = EnvConfig(**dict(CHECKPOINT=1))

        assert e3.check == 1

        e3.check = 2

        assert e3.check == 3


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner())
