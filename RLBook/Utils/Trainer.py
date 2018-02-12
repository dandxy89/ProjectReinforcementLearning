# -*- coding: utf-8 -*-
""" RLBook.Utils.Trainer

1.  Run Episode
2.  Training
3.  Concatenate into Batch

"""
from abc import abstractclassmethod, ABCMeta

from RLBook.Utils.Exceptions import IncorrectTrainerInputs
from RLBook.Utils.Player import Player


class Trainer:
    """ Base Class used for Interfacing with Agents and Environments
    """
    __metaclass__ = ABCMeta

    # class params
    EPISODE_MEM = [[], [], []]
    GAME = None
    AGENTS = []
    AGENT_GEN = None
    CONFIG = None

    CHECKPOINT = 0

    def __init__(self, environment, trainer_config):
        """ Initialise the Trainer Class

            :param environment:     Game or Env
            :param trainer_config:  All configuration settings for the Trainer

        """
        # Collect all the parameters
        self.GAME = environment
        self.CONFIG = trainer_config

        # Assert that the Trainer class is ready for action
        self.__validate()

    def __validate(self):
        """ Raise if none of the Core components have been loaded correctly
        """
        assert self.CONFIG is not None, IncorrectTrainerInputs("The Config has not been properly loaded")
        assert self.GAME is not None, IncorrectTrainerInputs("The Env / Game has not been properly loaded")

    def __reset_episode(self):
        self.GAME.reset()

    @abstractclassmethod
    def run_episode(self):
        """ While there exists legals moves within the Game, continue to play and collect all the states
        """
        pass

    @abstractclassmethod
    def self_play(self):
        pass

    @abstractclassmethod
    def dueling(self):
        pass

    @abstractclassmethod
    def human_play(self, human, agent: Player):
        pass

    @property
    def check(self):
        return self.CHECKPOINT

    @check.setter
    def check(self, x: int):
        self.CHECKPOINT += x

    def player_check(self, player: Player):
        return "{}{}".format(player.model_name, self.CHECKPOINT)

    def __repr__(self):
        return "< Base Class Trainer >"

    def __str__(self):
        return "< Base Class Trainer >"
