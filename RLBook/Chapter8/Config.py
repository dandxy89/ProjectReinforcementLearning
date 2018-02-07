# -*- coding: utf-8 -*-
""" Chapter8.KerasModel

-   Keras Model - Replicating the work by AlphaZero

"""
from datetime import datetime

from RLBook.Utils.PolicyTypes import PolicyEnum


class Config:
    """ Model Configuration
    """
    __acceptable_keys_list = ["MODEL_TYPE", "MODEL_NAME", "CNN_FILTER_NUM", "CNN_FILTER_SIZE", "VALUE_FC_SIZE",
                              "L2_REG", "RES_LAYER_NUM", "ACTIVATION_DENSE", "ACTIVATION", "N_LABELS",
                              "ACTIVATION_POLICY", "BATCH_SIZE", "EPOCHS"]
    RES_LAYER_NUM = 0
    ACTIVATION_DENSE = "tanh"
    ACTIVATION = "relu"
    L2_REG = 0.0002
    MODEL_TYPE = PolicyEnum.RESNET.value
    CNN_FILTER_NUM = 2
    CNN_FILTER_SIZE = 1
    MODEL_NAME = datetime.now().strftime("%Y%m%d_KerasModel_TTT")
    N_LABELS = 9
    ACTIVATION_POLICY = "softmax"
    VALUE_FC_SIZE = 1
    BATCH_SIZE = 1
    EPOCHS = 2

    def __init__(self, **kwargs):
        for k in kwargs.keys():
            if k in [self.__acceptable_keys_list]:
                self.__setattr__(k, kwargs[k])
