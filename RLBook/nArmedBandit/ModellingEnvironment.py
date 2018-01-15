# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.nArmedBandit.ModellingEnvironment

*   Capture results
*   Produce plots

"""
import logging

import numpy as np
import pandas as pd

from RLBook.nArmedBandit.Bandits import NArmBandit
from RLBook.nArmedBandit.EGreedy import EGreedy
from RLBook.nArmedBandit.Extras import MissingPolicyException, PolicyEnum
from RLBook.nArmedBandit.Incremental import Incremental
from RLBook.nArmedBandit.LinearRewardInaction import LinearInaction
from RLBook.nArmedBandit.LinearRewardPenalty import LinearPenalty
from RLBook.nArmedBandit.Nonstationary import Nonstationary
from RLBook.nArmedBandit.Pursuit import Pursuit
from RLBook.nArmedBandit.Softmax import Softmax

DEFAULT_TRIALS = 2000

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("nArmedBandits")


class ModelEnvironment:
    """
    """
    BANDIT_COUNT = None
    TRIAL_COUNT = DEFAULT_TRIALS
    POLICY = None
    BANDIT = None
    ALL_REWARDS = []
    POLICY_NAME = None
    POSITIVE_REWARDS = None
    IS_BINARY = False

    def __init__(self, bandits, trials=None, policy=None, epsilons=None, temperatures=None, alpha=None,
                 probability=0.4, beta=None):
        """

            :param bandits:
            :param trials:
            :param policy:
            :param epsilons:
            :param temperatures:

        """
        if trials is not None:
            self.TRIAL_COUNT = trials
        self.BANDIT_COUNT = bandits

        if policy is None:
            raise MissingPolicyException("Choose a Policy method!")
        else:
            self.POLICY_NAME = policy
            self.policy_selection(policy=policy, epsilons=epsilons, temperatures=temperatures,
                                  alpha=alpha, beta=beta)

        # Initialise a multi-Armed Bandit
        self.IS_BINARY = True if self.POLICY_NAME in PolicyEnum.BINARY_POLICIES else False
        self.BANDIT = NArmBandit(num=bandits,
                                 binary=self.IS_BINARY,
                                 probability=probability)

    def __repr__(self):
        return "< Modelling Environment Class >"

    def policy_selection(self, policy, epsilons=None, temperatures=None, alpha=None, beta=None):
        """ Policy Selection
        """
        if policy == PolicyEnum.EGREEDY:
            self.POLICY = EGreedy(num=self.BANDIT_COUNT, trials=self.TRIAL_COUNT,
                                  epsilon=epsilons)

        elif policy == PolicyEnum.SOFTMAX:
            self.POLICY = Softmax(num=self.BANDIT_COUNT, trials=self.TRIAL_COUNT,
                                  temperatures=temperatures)

        elif policy == PolicyEnum.LINEAR_REWARD_PENALTY:
            self.POLICY = LinearPenalty(num=self.BANDIT_COUNT, trials=self.TRIAL_COUNT,
                                        epsilon=epsilons, alpha=alpha)

        elif policy == PolicyEnum.LINEAR_REWARD_INACTION:
            self.POLICY = LinearInaction(num=self.BANDIT_COUNT, trials=self.TRIAL_COUNT,
                                         epsilon=epsilons, alpha=alpha)

        elif policy == PolicyEnum.INCREMENTAL:
            self.POLICY = Incremental(num=self.BANDIT_COUNT, trials=self.TRIAL_COUNT,
                                      epsilon=epsilons, alpha=alpha)

        elif policy == PolicyEnum.NON_STATIONARY:
            self.POLICY = Nonstationary(num=self.BANDIT_COUNT, trials=self.TRIAL_COUNT)

        elif policy == PolicyEnum.PURSUIT:
            self.POLICY = Pursuit(num=self.BANDIT_COUNT, trials=self.TRIAL_COUNT,
                                  epsilon=epsilons, beta=beta)

        else:
            raise NotImplementedError

    def run(self):
        """ TODO
        """
        for each_trial in range(self.TRIAL_COUNT):
            logger.debug("Running Trial: {}".format(each_trial + 1))
            if each_trial % 1000 == 0:
                logger.info("Running Trial: {}".format(each_trial + 1))

            # Get the Actions
            actions = self.POLICY.take_action(time=each_trial)

            # Get the Rewards
            rewards = [(each_epsilon_action, self.BANDIT.draw_bandit(index=each_epsilon_action))
                       for each_epsilon_action in actions]
            self.ALL_REWARDS.append(np.array([q for (p, q) in rewards]))

            # Update the Rewards
            self.POLICY.update_rewards(rewards=rewards, time=each_trial)

    def print_results(self):
        """ Display Results in the Console
        """
        # Positive Rewards
        self.POLICY.show_settings()
        self.POSITIVE_REWARDS = np.array(self.ALL_REWARDS) > 0
        print("Positive Rewards Achieved: ", self.POSITIVE_REWARDS.sum(axis=0))
        print("Positive Rewards Achieved (%): ", self.POSITIVE_REWARDS.sum(axis=0) / self.TRIAL_COUNT)

        # Show Weightings
        try:
            print("Show Weightings: {}".format(np.nan_to_num(self.POLICY.ACTION_REWARDS / self.POLICY.ACTION_COUNTS)))
        except:
            pass

    def generate_charts(self):
        """ Generating the Charts
        """
        import matplotlib.pyplot as plt

        if len(self.POSITIVE_REWARDS.shape) == 3:
            shape = self.POSITIVE_REWARDS.shape
            self.POSITIVE_REWARDS = self.POSITIVE_REWARDS.reshape((shape[0], shape[1]))

        cum_vals = np.cumsum(self.POSITIVE_REWARDS * 1, axis=0)
        rng = np.arange(start=1, stop=cum_vals.shape[0] + 1)
        optimal_results = pd.DataFrame(data=cum_vals / np.tile(rng, (len(self.POLICY.show_settings(p=False)), 1)).T,
                                       columns=[str(x) for x in self.POLICY.show_settings(p=False)])
        optimal_results["t"] = optimal_results.index + 1

        # Optimal Action Selection
        plt.figure()
        optimal_results.plot(x='t')
        plt.title("{} - Optimal Action (positive reward received).".format(self.POLICY_NAME))
        plt.ylabel("Optimal Action at t")
        plt.savefig("Plots/{}_Optimal_Action.png".format(self.POLICY_NAME))

        if not self.IS_BINARY:
            # Average Reward
            average_reward = pd.DataFrame(data=np.array(self.ALL_REWARDS),
                                          columns=[str(x) for x in self.POLICY.show_settings(p=False)])
            cols = average_reward.columns
            average_reward["t"] = average_reward.index + 1
            for each_column in cols:
                average_reward.loc[:, each_column] = average_reward.loc[:, each_column].cumsum() / average_reward.loc[:,
                                                                                                   "t"]

            plt.figure()
            average_reward.plot(x='t')
            plt.title("{} - Average Reward (positive reward received).".format(self.POLICY_NAME))
            plt.ylabel("Average Reward at t")
            plt.savefig("Plots/{}_Average_Reward.png".format(self.POLICY_NAME))

    def save(self):
        """ TODO
        """
        raise NotImplementedError
