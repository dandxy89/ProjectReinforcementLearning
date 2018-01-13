# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.nArmedBandit.Bandits

- 1-Arm Bandit
- n-Armed Bandit

"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


class OneArmBandit:
    """ A single slot machine
    """

    def __init__(self, mu, sigma):
        logger.debug("One Arm Bandit w Mean {mu} - Std {sigma}".format(mu=mu, sigma=sigma))
        self.mu, self.sigma = mu, sigma

    def __str__(self):
        return "One Arm Bandit with Mean {mu} - Std {sigma}".format(mu=self.mu, sigma=self.sigma)

    def __repr__(self):
        return "One Arm Bandit with Mean {mu} - Std {sigma}".format(mu=self.mu, sigma=self.sigma)

    def draw(self):
        return np.random.normal(self.mu, self.sigma)


class NArmBandit:
    """ A collection of N slot machines. Slot machine i has a mean of i and sigma of N
    """

    def __init__(self, num):
        self.num = num
        self.bandits = dict()

        for bandit_n in range(num):
            self.bandits[bandit_n] = OneArmBandit(mu=bandit_n, sigma=num)

    def __str__(self):
        return "{}-Arm Bandits".format(self.num)

    def __repr__(self):
        return "{}-Arm Bandits".format(self.num)

    def show(self):
        """ Display the Bandits (easier to view...)

            :return:            Dictionary of Bandits

        """
        return self.bandits

    def draw_bandit(self, index):
        """ Draw Bandit at Index

            :param index:       Index to draw upon (0 based index...)
            :return:            Bandit's Reward

        """
        return self.bandits[index].draw()
