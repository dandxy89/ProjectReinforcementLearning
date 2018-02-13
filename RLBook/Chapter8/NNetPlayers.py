#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.Chapter8.NNetPlayers
"""
from RLBook.Chapter8.Config import Config
from RLBook.Chapter8.KerasModel import KerasModel
from RLBook.Utils.Player import Player

# Default Players w Neural Networks
NNetPlayers = [Player(name='A', value=1, display='O', use_nn=True),
               Player(name='B', value=-1, display='X', use_nn=True)]


def create_keras_models(config1=Config(), config2=Config()):
    """ Create two Keras Models for either player specifically for the TicTacToe Game

        :param config1:     Configuration settings for player one
        :param config2:     Configuration settings for player two
        :return:            Dictionary of Two Keras Models

    """
    return {"1": KerasModel(config=config1), "-1": KerasModel(config=config2)}
