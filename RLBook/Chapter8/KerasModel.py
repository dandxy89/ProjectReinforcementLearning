# -*- coding: utf-8 -*-
""" Chapter8.KerasModel

-   Keras Model - Replicating the work by AlphaZero

"""
import numpy as np

from RLBook.Chapter8.Config import Config
from RLBook.Utils.NeuralNetwork import NeuralNet
from RLBook.Utils.PolicyTypes import PolicyEnum
from RLBook.Utils.ResNet import ResidualNet


class KerasModel(NeuralNet):
    """ Keras Model definition
    """

    def __init__(self, config: Config = Config()):
        """

            :param config:

        """
        super().__init__()

        if config.MODEL_TYPE == PolicyEnum.RESNET.value:
            self.net = ResidualNet(config=config)
            self.optimisation, self.model = self.net.compile_model()
        elif config.MODEL_TYPE == "LoadingExisting":
            self.net = "Awaiting load..."
        else:
            raise NotImplementedError

        self.config = config

    def __repr__(self):
        return "< Keras Model {} >".format(self.config.MODEL_NAME)

    def __str__(self):
        return "< Keras Model {} >".format(self.config.MODEL_NAME)

    @staticmethod
    def concatenate_arrays(tuple_arrays):
        """

            :param tuple_arrays:
            :return:

        """
        state_ary, policy_ary, z_ary = tuple_arrays[0], tuple_arrays[1], tuple_arrays[2]
        return np.concatenate(state_ary, axis=0), np.concatenate(policy_ary, axis=0), np.concatenate(z_ary, axis=0)

    def train(self, tuple_arrays):
        """

            :param tuple_arrays:
            :return:

        """
        # Concatenate the Arrays and get the training data
        state_ary, policy_ary, z_ary = self.concatenate_arrays(tuple_arrays=tuple_arrays)

        # Train the Model
        self.net.model.fit(x=state_ary, y=[policy_ary, z_ary], shuffle=self.config.EPOCHS,
                           batch_size=self.config.BATCH_SIZE, epochs=self.config.EPOCHS)

    def predict(self, tuple_arrays=None, state=None, current_player=0):
        """ Prediction

            :param tuple_arrays:    Handling a large batch
            :param state:           Handling a single state instance directly from the game
            :param current_player:  Current Player in index position
            :return:

        """
        if tuple_arrays is not None:
            tuples, value = self.net.model.predict(x=tuple_arrays, verbose=1)

            return [(val, prob) for val, prob in enumerate(tuples[0])], value[0][0]
        else:
            self._handle_state(state=state, current_player=current_player)

    def _handle_state(self, state, current_player):
        """

            :param state:
            :param current_player:
            :return:

        """
        raise NotImplementedError
