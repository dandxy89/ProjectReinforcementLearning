# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.nArmedBandit.Extras
"""


class MissingPolicyException(Exception):
    """ Raised if input data does not match expected schema
    """

    def __init__(self, msg):
        """ Initialise the Missing Policy Exception

            :param msg: message

        """
        self.msg = msg
        super(MissingPolicyException, self).__init__(msg)


class PolicyEnum:
    """
    """
    EGREEDY = "e-Greedy"
    SOFTMAX = "Softmax"
    LINEAR_REWARD_PENALTY = "Linear, reward-penalty"
    LINEAR_REWARD_INACTION = "Linear, reward-inaction"
    PURSUIT = "Pursuit"
    BINARY_POLICIES = [LINEAR_REWARD_INACTION, LINEAR_REWARD_PENALTY, PURSUIT]
    INCREMENTAL = "Incremental"
    NON_STATIONARY = "Nonstationary"

    def __repr__(self):
        return "< Policy Enum >"
