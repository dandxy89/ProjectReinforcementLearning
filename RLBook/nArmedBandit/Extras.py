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

    def __repr__(self):
        return "< Policy Enum >"
