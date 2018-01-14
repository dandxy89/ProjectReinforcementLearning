# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.nArmedBandit.Softmax

*   From: 2.3

"""
import numpy as np

DEFAULT_TEMPERATURE = [0.1, 0.3, 0.7]
np.random.seed(191989)


class Softmax:
    """ e-Greedy Policy Action
    """

    ACTION_REWARDS = None
    TEMPERATURES = DEFAULT_TEMPERATURE
    ACTIONS = None
    BANDITS = 1
    TRIAL_COUNT = 1
    TEMPERATURES_COUNT = 3

    def __init__(self, num, trials, temperatures=None):
        """ Initialise a Softmax Decision-maker

            :param num:                 Number of Bandits in use
            :param trials:              Number of Trials to run for
            :param temperatures:        List of Epsilons

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

    @staticmethod
    @np.vectorize
    def softmax_exp(a, temperature):
        """ Vectorized Exponential from the Softmax function

            :param a:
            :param temperature:
            :return:

        """
        return np.exp(a / temperature)

    def update_count(self, index, action):
        """ Increasing the Action count by 1

            :param index:
            :param action:
            :return:

        """
        self.ACTION_COUNTS[action, index] += 1

    def record_action(self, time, index, action):
        """ Record the Chosen action at time t

            :param time:
            :param index:
            :param action:
            :return:

        """
        self.ACTIONS[time, index] = action

    def take_action(self, time):
        """ Take an action for each Setting at time t

            :param time:
            :return:

        """
        actions = []
        for index, temperature in enumerate(self.TEMPERATURES):
            # Boltzmann Distribution
            choices = [a for a in range(self.N_BANDITS)]

            # Current Rewards
            boltz = self.softmax_exp(a=np.nan_to_num(self.ACTION_REWARDS[:, index] / self.ACTION_COUNTS[:, index]),
                                     temperature=temperature)

            # Select an Action
            action = np.random.choice(choices, p=boltz / boltz.sum())

            # Update the Count
            self.update_count(action=action, index=index)
            self.record_action(index=index, action=action, time=time)
            actions.append(action)

        return actions

    def update_reward(self, index, action, reward):
        """ Update a specific Reward Value

            :param index:
            :param action:
            :param reward:
            :return:

        """
        self.ACTION_REWARDS[action, index] += reward

    def update_rewards(self, rewards):
        """ Update all the Temperature Reward values

            :param rewards:
            :return:

        """
        for index, (action, reward) in enumerate(rewards):
            self.update_reward(index=index, action=action, reward=reward)

    def show_settings(self, p=True):
        if p:
            print(self.TEMPERATURES)
        return self.TEMPERATURES
