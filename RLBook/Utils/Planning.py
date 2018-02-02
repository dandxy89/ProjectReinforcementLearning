#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.Utils.Planning

-   Generic Class used for the Dynamic Programming (4) and Monte Carlo Methods (5) chapters since both
        chapters require in some form or another a 'rollout'.

"""
from abc import ABCMeta, abstractclassmethod


class BaseTreeNode(object):
    """ BaseTreeNode Class
    """
    PARENT = None
    CHILDREN = {}
    N_VISITS = 0

    __metaclass__ = ABCMeta

    def __init__(self, parent, prior_p: float):
        """ Initialise the Tree with Parent and Prior Probability

            :param parent:          Parent Node
            :param prior_p:         Prior Probability

        """
        self.PARENT = parent
        self.P = prior_p

    @abstractclassmethod
    def expand(self, *args):
        pass

    @abstractclassmethod
    def select(self, *args):
        pass

    @abstractclassmethod
    def update(self, *args):
        pass

    @abstractclassmethod
    def update_recursive(self, *args):
        pass

    @abstractclassmethod
    def get_value(self, *args):
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
