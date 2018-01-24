# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.Chapter2.Nonstationary

*   From: 2.6

"""
import numpy as np

from RLBook.Utils.PolicyTypes import PolicyEnum

DEFAULT_ALPHA = [0.1, 0.3, 0.7]
np.random.seed(191989)


class Nonstationary:
    """ Nonstationary Policy Agent
    """
    POLICY_TYPE = PolicyEnum.NON_STATIONARY
    ACTION_REWARDS = None
    ACTIONS = None
    BANDITS = 1
    TRIAL_COUNT = 1
    ALPHA_COUNT = 3
    ALPHA = DEFAULT_ALPHA
    EPSILON = None
    Q0 = None

    def __init__(self, num, trials, alpha=None, epsilon=0.1):
        """ Initialise a Nonstationary Policy

            :param num:         Number of Bandits in use
            :param trials:      Number of Trials to run for
            :param epsilon:     List of Epsilons

        """
        # params:
        self.N_BANDITS = num
        self.TRIAL_COUNT = trials
        self.EPSILON = epsilon
        self.Q0 = np.zeros((num, self.ALPHA_COUNT))

        # Using alternative epsilons
        if alpha is not None:
            self.ALPHA = alpha
            self.count_alphas()

        # Storage for Action Rewards
        self.ACTION_REWARDS = np.zeros(shape=(num, self.ALPHA_COUNT, trials))
        self.ACTION_REWARD = np.zeros(shape=(num, self.ALPHA_COUNT))
        self.ACTION_COUNTS = np.zeros(shape=(num, self.ALPHA_COUNT))

        # Storage for Action Take
        self.ACTIONS = np.zeros(shape=(trials, self.ALPHA_COUNT))

    def __repr__(self):
        return "< e-Greedy [nBandits {}, nTrials {}, alphas {}, epsilon {}] >".format(self.N_BANDITS,
                                                                                      self.TRIAL_COUNT,
                                                                                      self.ALPHA_COUNT,
                                                                                      self.EPSILON)

    def count_alphas(self):
        self.ALPHA_COUNT = len(self.ALPHA)

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
        return np.argmax(np.nan_to_num(self.ACTION_REWARD[:, index]))

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
        for index, alpha in enumerate(self.ALPHA):
            # Get the Greedy and Random Action
            choices = [self.greedy_action(index=index), self.random_action(index=index)]

            # Select an Action
            action = np.random.choice(choices, p=[1 - self.EPSILON, self.EPSILON])

            # Update the Count
            self.update_count(action=action, index=index)
            self.record_action(index=index, action=action, time=time)
            actions.append(action)

        return actions

    def update_rewards(self, rewards, time):
        """ Update all the Temperature Reward values

            :param rewards:
            :param time:
            :return:

        """
        for index, (action, reward) in enumerate(rewards):
            self.update_reward(index=index, action=action, reward=reward, time=time)

    def show_settings(self, p=True):
        if p:
            print(self.ALPHA)
        return self.ALPHA

    @staticmethod
    @np.vectorize
    def alpha_calculation(index, alpha):
        """ Alpha Calculation

            :param index:
            :param alpha:
            :return:

        """
        return alpha * np.power((1 - alpha), index)

    def update_reward(self, index, action, reward, time):
        """ Update a specific Reward Value

            :param index:
            :param action:
            :param reward:
            :param time:
            :return:

        """
        count = self.ACTION_COUNTS[action, index]
        alpha = self.ALPHA[index]
        self.ACTION_REWARDS[action, index, time] = reward

        if count == 0:
            # Update the Action
            self.ACTION_REWARD[action, index] = \
                np.power(1 - alpha, 2) * self.Q0[action, index] + alpha * (1 - alpha) * reward
        else:
            # Exponential, recency-weighted average
            rewards = self.ACTION_REWARDS[action, index, :]
            rewards = rewards[rewards != 0]
            # FIXME - V.slow implementation
            alphas_weights = self.alpha_calculation(index=np.arange(rewards.shape[0]), alpha=alpha)

            # Update the Action
            self.ACTION_REWARD[action, index] = np.multiply(rewards, alphas_weights).sum()
