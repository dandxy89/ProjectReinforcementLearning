# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" TicTacToe.Agent
"""
import uuid

import numpy as np

from RLBook.Utils.PolicyTypes import PolicyEnum


class RLAgent:
    """ Simple Epsilon Greedy Agent
    """
    POLICY_TYPE = PolicyEnum.EGREEDY
    EPSILON = 0.7
    ID = None
    ACTION = None
    POLICY = dict()
    STEP_SIZE = 0.6

    def __init__(self, epsilon: float = None, action_value: int = 1):
        if epsilon is not None:
            self.EPSILON = epsilon

        self.ACTION = action_value
        self.ID = str(uuid.uuid4())

    def __repr__(self):
        return "< e-Greedy Action Agent: e={} >".format(self.EPSILON)

    def __str__(self):
        return "< e-Greedy Action Agent: e={} >".format(self.EPSILON)

    def take_action(self, state: str, available_actions: np.ndarray) -> int:
        """ Take a e-Greedy Action

            :param state:                   Hashed State
            :param available_actions:       Available actions
            :return:                        Action

        """
        # Get the Greedy and Random Action
        choices = [self.greedy_action(state=state, available_actions=available_actions),
                   self.random_action(available_actions=available_actions)]

        # Select an Action
        action = np.random.choice(choices, p=[1 - self.EPSILON, self.EPSILON])

        return int(action)

    def update_policy(self, state: str, reward: float, action: int):
        """ Update the Policy given a State, Action and Reward

            :param state:                   Hashed State
            :param reward:                  Environment reward (t + 1)
            :param action:                  Action chosen

        """
        if state not in self.POLICY.keys():
            self.POLICY[state] = dict()
            self.POLICY[state][action] = reward

        else:
            if action not in self.POLICY[state].keys():
                self.POLICY[state][action] = reward

            else:
                self.POLICY[state][action] = (self.POLICY[state][action] \
                                              + self.STEP_SIZE * (reward - self.POLICY[state][action]))

    def random_action(self, available_actions: np.ndarray) -> int:
        """ Choose a Random Action

            :param available_actions:       Available actions
            :return:                        Chosen action

        """
        return np.random.choice(available_actions)

    def greedy_action(self, state: str, available_actions: np.ndarray) -> int:
        """ Greedy Epsilon Action

            :param state:                   Hashed State
            :param available_actions:       Available actions
            :return:                        Max value action

        """
        # Check if the State is in the Policy
        if state in self.POLICY.keys():
            if len([action for action in self.POLICY[state].keys() if action in available_actions.tolist()]) > 0:
                return max(action for action in self.POLICY[state].keys()
                           if action in available_actions.tolist())
            else:
                return self.random_action(available_actions=available_actions)
        # If not take a random action
        else:
            return self.random_action(available_actions=available_actions)
