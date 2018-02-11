# -*- coding: utf-8 -*-
""" Chapter8.KerasModel

-   Keras Model - Replicating the work by AlphaZero

"""
import datetime
import logging
import time

from RLBook.Utils.PolicyTypes import PolicyEnum


class Config:
    """ Model Configuration
    """
    __nn_keys_list = ["MODEL_TYPE", "MODEL_NAME", "CNN_FILTER_NUM", "CNN_FILTER_SIZE", "VALUE_FC_SIZE",
                      "L2_REG", "RES_LAYER_NUM", "ACTIVATION_DENSE", "ACTIVATION", "N_LABELS",
                      "ACTIVATION_POLICY", "BATCH_SIZE", "EPOCHS", "MCTS_ITERATIONS", "MCTS_MAX_TIME"]
    __mcts_keys_list = ["N_PLAYS", "N_WINS", "N_TIES", "SCORE", "PRIOR", "PRIOR",
                        "C_PUCT", "C_PUCT", "TAU", "Q", "U", "ACTION"]

    __acceptable_keys = __mcts_keys_list + __nn_keys_list

    # NNet Defaults
    RES_LAYER_NUM = 0
    ACTIVATION_DENSE = "tanh"
    ACTIVATION = "relu"
    L2_REG = 0.0002
    MODEL_TYPE = PolicyEnum.RESNET.value
    CNN_FILTER_NUM = 2
    CNN_FILTER_SIZE = 1
    MODEL_NAME = datetime.datetime.now().strftime("%Y%m%d_KerasModel_TTT")
    N_LABELS = 9
    ACTIVATION_POLICY = "softmax"
    VALUE_FC_SIZE = 1
    BATCH_SIZE = 1
    EPOCHS = 2
    MCTS_ITERATIONS = 30000
    MCTS_MAX_TIME = 20
    MOMENTUM = 0.9
    LR = 1e-2
    SHUFFLE = True

    # MCTS Defaults
    N_PLAYS = 0
    N_WINS = 0
    N_TIES = 0
    SCORE = 0.
    PRIOR = 1.
    C_PUCT = 2.
    TAU = 1.
    Q = 0.
    U = 0.
    ACTION = None
    TRAINING_MODE = True

    def __init__(self, **kwargs):
        """ Initialise the Config class

            :param kwargs:      Pass a dictionary where Keys are in the __acceptable_keys list

        """
        for k in kwargs.keys():
            # If the Key is in the accepted list then update
            if k in self.__acceptable_keys:
                self.__setattr__(k, kwargs[k])
            # Otherwise raise a warning
            else:
                logging.warning("Not adding: {}".format(k))

    @property
    def mcts_params(self):
        return {name: self.__getattribute__(name) for name in self.__mcts_keys_list}

    @property
    def mcts_search(self):
        return self.MCTS_ITERATIONS, self.MCTS_MAX_TIME

    @property
    def nn_params(self):
        return {name: self.__getattribute__(name) for name in self.__nn_keys_list}

    @property
    def get_all(self):
        return {name: self.__getattribute__(name) for name in self.__acceptable_keys}

    def __repr__(self):
        return "< Player Config >"


class EnvConfig:
    """ Environment Settings
    """
    __acceptable_keys = ["N_ITERATION", "N_EPISODE", "WIN_RATIO", "START_TIME",
                         "START_T", "EVALUATIONS", "CHECKPOINT"]

    def __init__(self, **kwargs):
        """

            :param kwargs:

        """
        # params
        self.N_ITERATION = 5
        self.N_EPISODE = 5
        self.WIN_RATIO = 0.55
        self.N_DUELS = 10
        self.START_TIME = datetime.datetime.now()
        self.START_T = time.time()
        self.EVALUATIONS = 10
        self.CHECKPOINT = 0

        for k in kwargs:
            # If the Key is in the accepted list then update
            if k in self.__acceptable_keys:
                self.__setattr__(k, kwargs[k])
            # Otherwise raise a warning
            else:
                logging.warning("Not adding: {}".format(k))

    def show_total_time(self):
        print("Been training for {}s".format(self.START_T - time.time()))

    def show_start_date(self):
        print(self.START_TIME)

    @property
    def get_all(self):
        return {name: self.__getattribute__(name) for name in self.__acceptable_keys}

    @property
    def check(self):
        return self.CHECKPOINT

    @check.setter
    def check(self, x):
        self.CHECKPOINT += x

    def __repr__(self):
        return "< Environment Config >"
