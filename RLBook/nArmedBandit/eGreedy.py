# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.nArmedBandit.eGreedy

*   From: 2.2

"""
import numpy as np

DEFAULT_EPSILON = [0, 0.01, 0.1]
np.random.seed(191989)


class eGreedy:
    """ e-Greedy Action Decision-making
    """

    ACTION_REWARDS = None
    EPSILON = DEFAULT_EPSILON
    ACTIONS = None
    BANDITS = 1
    TRIAL_COUNT = 1
    EPSILON_COUNT = 3

    def __init__(self, num, trials, epsilon=None):
        """ Initialisation an e-Greedy Decision-maker

            :param num:         Number of Bandits in use
            :param trials:      Number of Trials to run for
            :param epsilon:     List of Epsilons

        """
        # params:
        self.N_BANDITS = num
        self.TRIAL_COUNT = trials

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

    def random_action(self):
        return np.random.randint(self.N_BANDITS, size=1)

    def greedy_action(self, index):
        updated = np.nan_to_num(self.ACTION_REWARDS[:, index] / self.ACTION_COUNTS[:, index])
        return np.argmax(updated)

    def update_count(self, index, action):
        self.ACTION_COUNTS[action, index] += 1

    def record_action(self, time, index, action):
        self.ACTIONS[time, index] = action

    def take_action(self, time):
        actions = []
        for index, epsilon in enumerate(self.EPSILON):
            # Get the Greedy and Random Action
            choices = [self.greedy_action(index=index), self.random_action()]
            # Select an Action
            action = np.random.choice(choices, p=[1 - epsilon, epsilon])

            # Update the Count
            self.update_count(action=action, index=index)
            self.record_action(index=index, action=action, time=time)
            actions.append(action)

        return actions

    def update_reward(self, index, action, reward):
        self.ACTION_REWARDS[action, index] += reward

    def update_rewards(self, rewards):
        for index, (action, reward) in enumerate(rewards):
            self.update_reward(index=index, action=action, reward=reward)
