# -*- coding: utf-8 -*-
""" Chapter8.KerasModel

-   Keras Model - Replicating the work by AlphaZero

"""

from RLBook.Utils.PolicyTypes import PolicyEnum
from RLBook.Utils.ResNet import ResidualNet


class KerasModel(ResidualNet):
    """ Keras Model definition
    """

    def __init__(self, config):
        """

            :param config:

        """
        if config.MODEL_TYPE == PolicyEnum.RESNET.value:
            super().__init__(config=config)

        self.config = config
        self.optimisation, self.model = self.compile_model()
        self.check_point_counter = 0

    def train(self, tuple_arrays):
        """

            :param tuple_arrays:
            :return:

        """
        state_ary, policy_ary, z_ary = tuple_arrays[0], tuple_arrays[1], tuple_arrays[2]
        self.model.fit(x=state_ary, y=[policy_ary, z_ary],
                       batch_size=self.config.BATCH_SIZE, epochs=self.config.EPOCHS)

    def predict(self, tuple_arrays):
        """

            :param tuple_arrays:
            :return:

        """
        tuples, value = self.model.predict(x=tuple_arrays, verbose=1)
        return [(val, prob) for val, prob in enumerate(tuples[0])], value[0][0]
