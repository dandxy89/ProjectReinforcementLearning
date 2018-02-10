# -*- coding: utf-8 -*-
""" RLBook.Utiles.Trainer

1.  Run Episode
2.  Training
3.  Concatenate into Batch

"""
from abc import abstractclassmethod, ABCMeta

from RLBook.Utils.Exceptions import IncorrectTrainerInputs


class AllConfig:
    """ All Configuration setting for a Trainer
    """
    T = None

    def __repr__(self):
        return "< Env / Game & Trainer Configuration >"

    @property
    def trainer(self):
        return self.T

    @trainer.setter
    def trainer(self, value):
        self.T = value


class Trainer:
    """ Base Class used for Interfacing with Agents and Environments
    """
    __metaclass__ = ABCMeta

    # class params
    EPISODE_MEM = []
    GAME = None
    AGENTS = []
    AGENT_GEN = None
    ENV = None
    CONFIG = AllConfig()

    def __init__(self, environment, trainer_config):
        """ Initialise the Trainer Class

            :param environment:     Game or Env
            :param trainer_config:  All configuration settings for the Trainer

        """
        # Collect all the parameters
        self.ENV = environment
        self.CONFIG.trainer = trainer_config

        # Assert that the Trainer class is ready for action
        self.__validate()

    def __validate(self):
        """ Raise if none of the Core components have been loaded correctly
        """
        assert len(self.AGENTS) != 0 or self.AGENTS is not None, \
            IncorrectTrainerInputs("The Agent(s) have not been properly loaded")
        assert self.CONFIG is not None, IncorrectTrainerInputs("The Config has not been properly loaded")
        assert self.ENV is not None, IncorrectTrainerInputs("The Env / Game has not been properly loaded")

    def __reset_episode(self):
        self.ENV.reset()

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
    def human_play(self):
        pass
