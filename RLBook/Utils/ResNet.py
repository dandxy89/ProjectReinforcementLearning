# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.Utils.ResNet
"""
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, Add
from keras.optimizers import SGD
from keras.regularizers import l2

from RLBook.Utils.NeuralNetwork import objective_function_for_policy, objective_function_for_value


class ResidualNet:
    """ ResNet Construction
    """

    def __init__(self, config):
        self.config = config
        self.optimisation = None

    def build(self):
        """ Construct the whole Neural Network

            :return:    Keras Model
            :rtype:     Model

        """
        mc = self.config
        in_x = x = Input((2, 3, 3))

        # Where input is (batch, channels, height, width)
        x = Conv2D(filters=mc.CNN_FILTER_NUM, kernel_size=mc.CNN_FILTER_SIZE, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.L2_REG))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation(mc.ACTIVATION)(x)

        if mc.RES_LAYER_NUM > 0:
            for _ in range(mc.RES_LAYER_NUM):
                x = self._build_residual_block(x)

        res_out = x
        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", kernel_regularizer=l2(mc.L2_REG))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation(mc.ACTIVATION)(x)
        x = Flatten()(x)
        # no output for 'pass'
        policy_out = Dense(self.config.N_LABELS, kernel_regularizer=l2(mc.L2_REG), activation=mc.ACTIVATION_POLICY,
                           name="policy_output")(x)

        # for value output
        x = Conv2D(filters=1, kernel_size=1, data_format="channels_first", kernel_regularizer=l2(mc.L2_REG))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation(mc.ACTIVATION)(x)
        x = Flatten()(x)
        x = Dense(mc.VALUE_FC_SIZE, kernel_regularizer=l2(mc.L2_REG), activation=mc.ACTIVATION)(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.L2_REG),
                          activation=mc.ACTIVATION_DENSE, name="value_output")(x)

        # Collect into one Model
        return Model(in_x, [policy_out, value_out], name=mc.MODEL_NAME)

    def _build_residual_block(self, x):
        """ Add a sequence of the Residual Block

            :param x:       An existing Keras Model
            :return:        Appended Residual Block onto the existing Model
            :rtype:         Model

        """
        mc = self.config
        in_x = x
        x = Conv2D(filters=mc.CNN_FILTER_NUM, kernel_size=mc.CNN_FILTER_SIZE, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.L2_REG))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation(mc.ACTIVATION)(x)
        x = Conv2D(filters=mc.CNN_FILTER_NUM, kernel_size=mc.CNN_FILTER_SIZE, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.L2_REG))(x)
        x = BatchNormalization(axis=1)(x)
        x = Add()([in_x, x])
        x = Activation(mc.ACTIVATION)(x)

        return x

    def compile_model(self):
        """ Compile the Keras Model

            :return:        Keras compiled model and optimisation
            :rtype:         Optimizer, Model

        """
        model = self.build()
        # Define the Optimizer
        optimisation = SGD(lr=self.config.LR,
                           momentum=self.config.MOMENTUM)

        # Compile the Duel headed NNet
        model.compile(optimizer=optimisation,
                      loss=[objective_function_for_policy, objective_function_for_value])

        return optimisation, model
