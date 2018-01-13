# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.nArmedBandit.ModellingEnvironment

*   Capture results
*   Produce plots

"""
import numpy as np

from RLBook.nArmedBandit.Bandits import NArmBandit
from RLBook.nArmedBandit.Extras import MissingPolicyException, PolicyEnum
from RLBook.nArmedBandit.Softmax import Softmax
from RLBook.nArmedBandit.eGreedy import eGreedy

DEFAULT_TRIALS = 2000


class ModelEnvironment:
    """
    """
    BANDIT_COUNT = None
    TRIAL_COUNT = DEFAULT_TRIALS
    POLICY = None
    BANDIT = None
    ALL_REWARDS = []
    POLICY_NAME = None

    def __init__(self, bandits, trials=None, policy=None):
        """

            :param bandits:
            :param trials:
            :param policy:

        """
        if trials is not None:
            self.TRIAL_COUNT = trials
        self.BANDIT_COUNT = bandits

        if policy is None:
            raise MissingPolicyException("Choose a Policy method!")
        else:
            self.POLICY_NAME = policy
            self.policy_selection(policy=policy)

        # Initialise a multi-Armed Bandit
        self.BANDIT = NArmBandit(num=bandits)

    def __repr__(self):
        return "< Modelling Environment Class >"

    def policy_selection(self, policy):
        """
        """
        if policy == PolicyEnum.EGREEDY:
            self.POLICY = eGreedy(num=self.BANDIT_COUNT, trials=self.TRIAL_COUNT)
        elif policy == PolicyEnum.SOFTMAX:
            self.POLICY = Softmax(num=self.BANDIT_COUNT, trials=self.TRIAL_COUNT)
        else:
            raise NotImplementedError

    def run(self):
        """
        """
        for each_trial in range(self.TRIAL_COUNT):
            print("Running Trial: {}".format(each_trial + 1))
            # Get the Actions
            actions = self.POLICY.take_action(time=each_trial)

            # Get the Rewards
            rewards = [(each_epsilon_action, self.BANDIT.draw_bandit(index=each_epsilon_action))
                       for each_epsilon_action in actions]
            self.ALL_REWARDS.append(np.argmax(np.array(rewards)[:, 1]))

            # Update the Rewards
            self.POLICY.update_rewards(rewards=rewards)

    def print_results(self):
        """
        """
        optimal = np.array(self.ALL_REWARDS)
        ua, uind = np.unique(optimal, return_inverse=True)
        count = np.bincount(uind)

        if self.POLICY_NAME == PolicyEnum.EGREEDY:
            print("Optimal Selection: {}".format(list(zip(self.POLICY.EPSILON, count / optimal.shape[0]))))
            print("Average Reward: {}".format(np.average(self.POLICY.ACTION_REWARDS, axis=0) / self.TRIAL_COUNT))
            print(np.nan_to_num(self.POLICY.ACTION_REWARDS / self.POLICY.ACTION_COUNTS))

        elif self.POLICY_NAME == PolicyEnum.SOFTMAX:
            print("Optimal Selection: {}".format(list(zip(self.POLICY.TEMPERATURES, count / optimal.shape[0]))))
            print("Average Reward: {}".format(np.average(self.POLICY.ACTION_REWARDS, axis=0) / self.TRIAL_COUNT))
            print(np.nan_to_num(self.POLICY.ACTION_REWARDS / self.POLICY.ACTION_COUNTS))

        else:
            raise ModuleNotFoundError

    def generate_charts(self):
        """
        """
        raise NotImplementedError

    def save(self):
        """
        """
        raise NotImplementedError


if __name__ == '__main__':
    # params:
    bandits = 10
    trials = 5000

    # Initialise an Environment
    env1 = ModelEnvironment(trials=trials, bandits=bandits, policy=PolicyEnum.EGREEDY)
    env2 = ModelEnvironment(trials=trials, bandits=bandits, policy=PolicyEnum.SOFTMAX)
    env1.run()
    env2.run()

    # Show results
    print("e-Greedy Policy")
    env1.print_results()
    print("\n\nSoftmax Policy")
    env2.print_results()

