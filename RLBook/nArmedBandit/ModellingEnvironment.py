# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.nArmedBandit.ModellingEnvironment

*   Capture results
*   Produce plots

"""
import logging

import numpy as np

from RLBook.nArmedBandit.Bandits import NArmBandit
from RLBook.nArmedBandit.EGreedy import EGreedy
from RLBook.nArmedBandit.Extras import MissingPolicyException, PolicyEnum
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

    def __init__(self, bandits, trials=None, policy=None, epsilons=None, temperatures=None):
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
            self.policy_selection(policy=policy, epsilons=epsilons, temperatures=temperatures)

        # Initialise a multi-Armed Bandit
        self.BANDIT = NArmBandit(num=bandits)

    def __repr__(self):
        return "< Modelling Environment Class >"

    def policy_selection(self, policy, epsilons=None, temperatures=None):
        """ Policy Selection
        """
        if policy == PolicyEnum.EGREEDY:
            self.POLICY = EGreedy(num=self.BANDIT_COUNT, trials=self.TRIAL_COUNT, epsilon=epsilons)

        elif policy == PolicyEnum.SOFTMAX:
            self.POLICY = Softmax(num=self.BANDIT_COUNT, trials=self.TRIAL_COUNT, temperatures=temperatures)

        else:
            raise NotImplementedError

    def run(self):
        """ TODO
        """
        for each_trial in range(self.TRIAL_COUNT):
            logger.debug("Running Trial: {}".format(each_trial + 1))
            if each_trial % 100 == 0:
                logger.info("Running Trial: {}".format(each_trial + 1))

            # Get the Actions
            actions = self.POLICY.take_action(time=each_trial)

            # Get the Rewards
            rewards = [(each_epsilon_action, self.BANDIT.draw_bandit(index=each_epsilon_action))
                       for each_epsilon_action in actions]
            self.ALL_REWARDS.append(np.array([q for (p, q) in rewards]))

            # Update the Rewards
            self.POLICY.update_rewards(rewards=rewards)

    def print_results(self):
        """ TODO
        """
        # Positive Rewards
        self.POLICY.show_settings()
        print("Positive Rewards: ", (np.array(self.ALL_REWARDS) > 0).sum(axis=0))

        # Show Weightings
        print("Show Weightings: ")
        print(np.nan_to_num(self.POLICY.ACTION_REWARDS / self.POLICY.ACTION_COUNTS))

    def generate_charts(self):
        """ TODO
        """
        raise NotImplementedError

    def save(self):
        """ TODO
        """
        raise NotImplementedError


if __name__ == '__main__':
    # params:
    bandits = 10
    trials = 20000

    # Initialise a e-Greedy Environment
    # env1 = ModelEnvironment(trials=trials, bandits=bandits, policy=PolicyEnum.EGREEDY,
    #                         epsilons=[0, 0.01, 0.1, 0.3, 0.5, 0.75, 0.95])
    # env1.run()
    # env1.print_results()
    # @20000 trials:    Positive Rewards: [13147 15426 15981]
    #                   Epsilon: [0, 0.01, 0.1]
    # @20000 trials:    Positive Rewards: [10071 16223 15948 15031 14827 13704 13240]
    #                   Epsilon: [0, 0.01, 0.1, 0.3, 0.5, 0.75, 0.95]

    # Initialise a Softmax Environment
    env2 = ModelEnvironment(trials=trials, bandits=bandits, policy=PolicyEnum.SOFTMAX,
                            temperatures=[0.0000000000001, 0.01, 0.1, 0.3, 0.5, 0.75, 0.95])
    env2.run()
    env2.print_results()
    # @20000 trials:    Positive Rewards:  [13133 15778 16239]
    #                   Temps [0.1, 0.3, 0.7]
    # @20000 trials:    Positive Rewards: [ 9897  9948 12998 14534 15733 15485 16338]
    #                   Temps [0.0000000000001, 0.01, 0.1, 0.3, 0.5, 0.75, 0.95]
