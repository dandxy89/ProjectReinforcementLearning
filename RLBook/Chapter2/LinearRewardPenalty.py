# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.Chapter2.LinearRewardPenalty

*   From: 2.4

Notes:

*   Technically this is for the Binary version

"""
import numpy as np

from RLBook.Utils.PolicyTypes import PolicyEnum

DEFAULT_EPSILON = [0, 0.01, 0.1]
DEFAULT_ALPHA = 0.1
np.random.seed(191989)


class LinearPenalty:
    """ Linear, reward-penalty Policy Action
    """
    POLICY_TYPE = PolicyEnum.LINEAR_REWARD_PENALTY
    ACTION_REWARDS = None
    EPSILON = DEFAULT_EPSILON
    ACTIONS = None
    BANDITS = 1
    TRIAL_COUNT = 1
    EPSILON_COUNT = 3
    ALPHA = DEFAULT_ALPHA

    def __init__(self, num, trials, epsilon=None, alpha=DEFAULT_ALPHA):
        """ Initialise a Linear, reward-penalty Policy

            :param num:         Number of Bandits in use
            :param trials:      Number of Trials to run for
            :param epsilon:     List of Epsilons
            :param alpha:       Set 'alpha'

        """
        # params:
        self.N_BANDITS = num
        self.TRIAL_COUNT = trials
        if alpha != DEFAULT_ALPHA and alpha is not None:
            self.ALPHA = alpha

        # Using alternative epsilons
        if epsilon is not None:
            self.EPSILON = epsilon
            self.count_epsilons()

        # Storage for Action Rewards
        self.ACTION_REWARDS = np.zeros(shape=(num, self.EPSILON_COUNT))
        self.ACTION_COUNTS = np.zeros(shape=(num, self.EPSILON_COUNT))

        # Storage for Action Take
        self.ACTIONS = np.zeros(shape=(trials, self.EPSILON_COUNT))

    def __repr__(self):
        return "< e-Greedy [nBandits {}, nTrials {}, epsilons {}] >".format(self.N_BANDITS,
                                                                            self.TRIAL_COUNT,
                                                                            self.EPSILON)

    def count_epsilons(self):
        self.EPSILON_COUNT = len(self.EPSILON)

    def random_action(self, index):
        a = np.random.randint(self.N_BANDITS, size=1)

        # If the Action is the same as the Greedy Action attempt to get another... exploration
        if a == self.greedy_action(index=index):
            return np.random.randint(self.N_BANDITS, size=1)
        # Otherwise
        else:
            return a

    def greedy_action(self, index):
        """ Take a Greedy Action

            :param index:
            :return:

        """
        return np.argmax(np.nan_to_num(self.ACTION_REWARDS[:, index] / self.ACTION_COUNTS[:, index]))

    def update_count(self, index, action):
        """ Increase the Action Count when used and log the result

            :param index:
            :param action:
            :return:

        """
        self.ACTION_COUNTS[action, index] += 1

    def record_action(self, time, index, action):
        """ Log the Action in an Array

            :param time:
            :param index:
            :param action:
            :return:

        """
        self.ACTIONS[time, index] = action

    def take_action(self, time):
        """ Take an Action using the Policies

            :param time:
            :return:

        """
        actions = []
        for index, epsilon in enumerate(self.EPSILON):
            # Get the Greedy and Random Action
            choices = [self.greedy_action(index=index), self.random_action(index=index)]
            # Select an Action
            action = np.random.choice(choices, p=[1 - epsilon, epsilon])

            # Update the Count
            self.update_count(action=action, index=index)
            self.record_action(index=index, action=action, time=time)
            actions.append(action)

        return actions

    def update_reward(self, index, action, reward):
        """ Update a specific Reward Value

            INFO:

                Changed update rule

            :param index:
            :param action:
            :param reward:
            :return:

        """
        current = self.ACTION_REWARDS[action, index]
        self.ACTION_REWARDS[action, index] = current + self.ALPHA * (1 - current)

    def update_rewards(self, rewards):
        """ Update all the Temperature Reward values

            :param rewards:
            :return:

        """
        for index, (action, reward) in enumerate(rewards):
            self.update_reward(index=index, action=action, reward=reward)

    def show_settings(self, p=True):
        if p:
            print(self.EPSILON)
        return self.EPSILON
