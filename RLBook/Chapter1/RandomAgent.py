# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" Chapter1.Agent
"""

import numpy as np

from RLBook.Utils.PolicyTypes import PolicyEnum


class RLAgentRandom:
    """ Simple Epsilon Greedy Agent
    """
    POLICY_TYPE = PolicyEnum.RANDOM
    ID = 2
    ACTION = None
    POLICY = dict()
    STORAGE = []

    def __init__(self, action_value: int = 1):
        self.ACTION = action_value

    def __repr__(self):
        return "< Random Action Agent >"

    def __str__(self):
        return "< Random Action Agent >"

    def append(self, state, reward, action):
        """ Append in the Storage variable

            :param state:       Game state
            :param reward:      Reward value

        """
        self.STORAGE.append((state, action, reward))

    def reset(self):
        """ Reset the Storage

            :return:

        """
        self.STORAGE = []

    def take_action(self, state: str, available_actions: np.ndarray) -> int:
        """ Take a e-Greedy Action

            :param state:                   Hashed State
            :param available_actions:       Available actions
            :return:                        Action

        """
        # Select an Action
        action = self.random_action(available_actions=available_actions)

        return int(action)

    def update_policy(self):
        """ Update the Policy given a State, Action and Reward
        """
        pass

    def random_action(self, available_actions: np.ndarray) -> int:
        """ Choose a Random Action

            :param available_actions:       Available actions
            :return:                        Chosen action

        """
        return np.random.choice(available_actions)
