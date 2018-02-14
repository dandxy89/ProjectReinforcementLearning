# -*- coding: utf-8 -*-
""" Chapter8.KerasModel

-   Keras Model - Replicating the work by AlphaZero

"""
import logging

import numpy as np

from RLBook.Chapter8.Config import Config
from RLBook.Utils.NeuralNetwork import NeuralNet
from RLBook.Utils.PolicyTypes import PolicyEnum
from RLBook.Utils.ResNet import ResidualNet


class KerasModel(NeuralNet):
    """ Keras Model definition
    """

    def __init__(self, config: Config = Config()):
        """ Initialise a Keras Model with its own configuration

            Default will not include any ResNet blocks.

            :param config:

        """
        super().__init__()

        if config.MODEL_TYPE == PolicyEnum.RESNET.value:
            self.net = ResidualNet(config=config)
            self.optimisation, self.model = self.net.compile_model()

        elif config.MODEL_TYPE == "LoadingExisting":
            logging.warning("Awaiting load...")
            self.net = "Awaiting load..."

        else:
            logging.warning("Neural Network selection not known.")
            raise NotImplementedError

        self.config = config

    def __repr__(self):
        return "< Keras Model {} >".format(self.config.MODEL_NAME)

    def __str__(self):
        return "< Keras Model {} >".format(self.config.MODEL_NAME)

    @staticmethod
    def concatenate_arrays(tuple_arrays):
        """ This will translate the List of Lists into three numpy arrays

            :param tuple_arrays:    List of Lists containing the training data
            :return:                Three numpy arrays for State, Policy and Value

        """
        state_ary, policy_ary, z_ary = tuple_arrays[0], tuple_arrays[1], tuple_arrays[2]
        return np.concatenate(state_ary, axis=0), \
               np.concatenate(policy_ary, axis=0), \
               np.concatenate(z_ary, axis=0)

    def train(self, tuple_arrays):
        """ Training method to invoke the training of the Neural Network

            :param tuple_arrays:    List of Lists containing the training data
            :return:                None

        """
        # Concatenate the Arrays and get the training data
        state_ary, policy_ary, z_ary = self.concatenate_arrays(tuple_arrays=tuple_arrays)

        try:
            # Train the Model
            self._train(policy_ary, state_ary, z_ary)

        except RuntimeError:
            # The Model may require it to be compiled post load
            self.model.compile()

            # Train the Model
            self._train(policy_ary, state_ary, z_ary)

    def _train(self, policy_ary, state_ary, z_ary):
        """ Private method to call the training of the model

            :param policy_ary:      Policy
            :param state_ary:       State Matrix
            :param z_ary:           Value

        """
        # Train the Model
        self.model.fit(x=state_ary, y=[policy_ary, z_ary],
                       shuffle=self.config.EPOCHS,
                       batch_size=self.config.BATCH_SIZE,
                       epochs=self.config.EPOCHS,
                       verbose=self.config.VERBOSE)

    def predict(self, tuple_arrays=None, state=None, current_player=0):
        """ Prediction

            :param tuple_arrays:    Handling a large batch
            :param state:           Handling a single state instance directly from the game
            :param current_player:  Current Player in index position
            :return:

        """
        tuples, value = self.model.predict(x=tuple_arrays, verbose=self.config.VERBOSE)

        return [(val, prob) for val, prob in enumerate(tuples[0])], value[0][0]

    def _handle_state(self, state, current_player):
        """

            :param state:
            :param current_player:
            :return:

        """
        raise NotImplementedError
