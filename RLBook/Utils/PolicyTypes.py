# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.Utils.PolicyTypes
"""
from enum import Enum


class PolicyEnum(Enum):
    """ Policy Enumerator
    """
    EGREEDY = "e-Greedy"
    SOFTMAX = "Softmax"
    LINEAR_REWARD_PENALTY = "Linear, reward-penalty"
    LINEAR_REWARD_INACTION = "Linear, reward-inaction"
    PURSUIT = "Pursuit"
    BINARY_POLICIES = [LINEAR_REWARD_INACTION, LINEAR_REWARD_PENALTY, PURSUIT]
    INCREMENTAL = "Incremental"
    NON_STATIONARY = "Nonstationary"
    RANDOM = "Random"
    UCB = "Upper Confidence Bound"
