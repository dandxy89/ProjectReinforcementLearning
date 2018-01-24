# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" TicTacToe.Agent
"""

import numpy as np

from RLBook.TicTacToe.RandomAgent import RLAgentRandom
from RLBook.Utils.PolicyTypes import PolicyEnum


class RLAgent(RLAgentRandom):
    """ Simple Epsilon Greedy Agent with a Discounted Policy Update
    """
    POLICY_TYPE = PolicyEnum.EGREEDY
    EPSILON = 0.45
    ID = 1
    ACTION = None
    POLICY = dict()
    STEP_SIZE = 0.6
    GAMMA = 0.5
    STORAGE = []
    UPDATE = True

    def __init__(self, epsilon: float = None, action_value: int = 1, step_size: float = None):
        super().__init__()
        if epsilon is not None:
            self.EPSILON = epsilon

        if step_size is not None:
            self.STEP_SIZE = step_size

        self.ACTION = action_value

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

    def update_policy(self):
        """ Update the Policy given a State, Action and Reward
        """
        if self.UPDATE:
            # Initialise the params:
            update_value, reward = [], self.STORAGE[-1][2]

            # Apply a discounting
            for (state, action, _), value in zip(self.STORAGE,
                                                 np.arange(start=len(self.STORAGE), stop=0, step=-1)):
                # Adjusted Weighted
                reward = np.power((1 - self.GAMMA), value) * reward
                update_value.append(reward)

            # Calculate the Rewards
            reward = np.array(update_value).sum()

            for (state, action, _), value in zip(self.STORAGE,
                                                 np.arange(start=len(self.STORAGE), stop=0, step=-1)):
                # Update the Policy
                if state not in self.POLICY.keys():
                    self.POLICY[state] = dict()
                    self.POLICY[state][action] = reward + np.random.random() / 100000000

                else:
                    if action not in self.POLICY[state].keys():
                        self.POLICY[state][action] = reward + np.random.random() / 100000000

                    else:
                        self.POLICY[state][action] = (self.POLICY[state][action] +
                                                      self.STEP_SIZE * (reward - self.POLICY[state][action]))

    def greedy_action(self, state: str, available_actions: np.ndarray) -> int:
        """ Greedy Epsilon Action

            :param state:                   Hashed State
            :param available_actions:       Available actions
            :return:                        Max value action

        """
        # Check if the State is in the Policy
        if state in self.POLICY.keys():
            if len([action for action in self.POLICY[state].keys() if action in available_actions.tolist()]) > 0:
                # Rearrange the Max
                inverse = [(value, key) for key, value in self.POLICY[state].items() if
                           key in available_actions.tolist()]
                return max(inverse)[1]
            else:
                return self.random_action(available_actions=available_actions)

        # If not take a random action
        else:
            return self.random_action(available_actions=available_actions)
