#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.Utils.Planning

-   Generic Class used for the Dynamic Programming (4) and Monte Carlo Methods (5) chapters since both
        chapters require in some form or another a 'rollout'.

"""
import numpy as np


class BaseTreeNode(object):
    """ BaseTreeNode Class
    """
    PARENT = None
    CHILDREN = {}
    N_VISITS = 0

    def __init__(self, parent, prior_p: float):
        """ Initialise the Tree with Parent and Prior Probability

            :param parent:          Parent Node
            :param prior_p:         Prior Probability

        """
        self.PARENT = parent
        self.P = prior_p

    def expand(self, action_priors: np.ndarray):
        pass

    def select(self, c_puct: float):
        pass

    def update(self, leaf_value: float):
        pass

    def update_recursive(self, leaf_value):
        pass

    def get_value(self, c_puct: float):
        pass

    def is_leaf(self) -> bool:
        """ Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self.CHILDREN == {}

    def is_root(self) -> bool:
        return self.CHILDREN is None

    def __repr__(self) -> str:
        return "< BaseTreeNode Parent {} with Children {} >".format(self.PARENT, self.CHILDREN)

    def __str__(self) -> str:
        return "< BaseTreeNode Parent {} with Children {} >".format(self.PARENT, self.CHILDREN)
