#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Testing for Chapter 8 Config
"""
import datetime
import unittest

from RLBook.Chapter8.Config import Config
from RLBook.Chapter8.KerasModel import KerasModel


class TestChapter8KerasModel(unittest.TestCase):
    """ Testing the Chapter8 Implementations
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_lazy_init(self):
        """ TODO
        """
        c = Config()
        c.MODEL_TYPE = "LoadingExisting"
        km = KerasModel(config=c)

        assert km.increment == 0
        assert km.__repr__() == "< Keras Model {}_KerasModel_TTT >".format(datetime.datetime.now().strftime("%Y%m%d"))
        assert km.__str__() == "< Keras Model {}_KerasModel_TTT >".format(datetime.datetime.now().strftime("%Y%m%d"))
        assert km.net == "Awaiting load..."

    def test_resnet_init(self):
        """ TODO
        """
        km2 = KerasModel()

        assert km2.model.metrics_names == ['loss', 'policy_output_loss', 'value_output_loss']
        assert km2.model.count_params() == 218

    def test_resnet_one_block(self):
        """ TODO
        """
        c3 = Config()
        c3.RES_LAYER_NUM = 1
        km3 = KerasModel(config=c3)

        assert km3.model.count_params() == 246


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner())
