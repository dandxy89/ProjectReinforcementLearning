# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.nArmedBandit.Softmax

*   From: 2.3

"""
import numpy as np

DEFAULT_TEMPERATURE = [0.1, 0.3, 0.7]
np.random.seed(191989)


class Softmax:
    """ e-Greedy Action Decision-making
    """

    ACTION_REWARDS = None
    TEMPERATURES = DEFAULT_TEMPERATURE
    ACTIONS = None
    BANDITS = 1
    TRIAL_COUNT = 1
    TEMPERATURES_COUNT = 3

    def __init__(self, num, trials, temperatures=None):
        """ Initialise a Softmax Decision-maker

            :param num:         Number of Bandits in use
            :param trials:      Number of Trials to run for
            :param epsilon:     List of Epsilons

        """
        # params:
        self.N_BANDITS = num
        self.TRIAL_COUNT = trials

        # Using alternative epsilons
        if temperatures is not None:
            self.TEMPERATURES = temperatures
            self.count_temperatures()

        # Storage for Action Rewards
        self.ACTION_REWARDS = np.zeros(shape=(num, self.TEMPERATURES_COUNT))
        self.ACTION_COUNTS = np.zeros(shape=(num, self.TEMPERATURES_COUNT))

        # Storage for Action Take
        self.ACTIONS = np.zeros(shape=(trials, self.TEMPERATURES_COUNT))

    def __repr__(self):
        return "< e-Greedy [nBandits {}, nTrials {}, epsilons {}] >".format(self.N_BANDITS,
                                                                            self.TRIAL_COUNT,
                                                                            self.TEMPERATURES)

    def count_temperatures(self):
        self.TEMPERATURES_COUNT = len(self.TEMPERATURES)

    def random_action(self):
        return np.random.randint(self.N_BANDITS, size=1)

    def get_value(self, a, index):
        return np.nan_to_num(self.ACTION_REWARDS[a, index] / self.ACTION_COUNTS[a, index])

    def softmax_exp(self, a, temperature, index):
        return np.exp(self.get_value(a=a, index=index) / temperature)

    def boltzmann_distribution(self, action, temperature, index):
        return self.softmax_exp(action, temperature, index=index) / sum(
            [self.softmax_exp(a=a, temperature=temperature, index=index) for a in range(self.N_BANDITS)])

    def update_count(self, index, action):
        self.ACTION_COUNTS[action, index] += 1

    def record_action(self, time, index, action):
        self.ACTIONS[time, index] = action

    def take_action(self, time):
        actions = []
        for index, temperature in enumerate(self.TEMPERATURES):
            # Boltzmann Distribution
            choices = [a for a in range(self.N_BANDITS)]
            p = [self.boltzmann_distribution(action=a, temperature=temperature, index=index)
                 for a in range(self.N_BANDITS)]

            # Select an Action
            action = np.random.choice(choices, p=p)

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
