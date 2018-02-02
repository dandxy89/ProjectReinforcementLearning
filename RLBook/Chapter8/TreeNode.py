# -*- coding: utf-8 -*-
""" Chapter5.Monte Carlo Tree Search

Monte Carlo Tree Search

"""

import numpy as np

from RLBook.Utils.Planning import BaseTreeNode


class TreeNode(BaseTreeNode):
    """ TreeNode Class

        A node in the MCTS tree. Each node keeps track of its own value Q, prior probability P, and
        its visit-count-adjusted prior score u.

    """
    P = None
    Q = 0
    U = 0

    def __init__(self, parent, prior_p: float):
        """ Initialise the Tree with Parent and Prior Probability

            :param parent:
            :param prior_p:

        """
        super(TreeNode, self).__init__(parent=parent, prior_p=prior_p)

    def expand(self, action_priors: np.ndarray):
        """ Expand tree by creating new children.

            :param action_priors:       output from policy function - a list of tuples of actions
                                        and their prior probability according to the policy function

        """
        for action, prob in action_priors:
            if action not in self.CHILDREN:
                self.CHILDREN[action] = TreeNode(self, prob)

    def select(self, c_puct: float) -> float:
        """ Select action among children that gives maximum action value, Q plus bonus u(P).

            :param c_puct:      a number in (0, inf) controlling the relative impact of values, Q, and prior
                                probability, P, on this node's score.
            :return:            A tuple of (action, next_node)

        """
        return max(self.CHILDREN.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value: float):
        """ Update node values from leaf evaluation.

            :param leaf_value:      the value of subtree evaluation from the current player's perspective

        """
        # Count visit.
        self.N_VISITS += 1

        # Update Q, a running average of values for all visits.
        self.Q += 1.0 * (leaf_value - self.Q) / self.N_VISITS

    def update_recursive(self, leaf_value: float):
        """ Like a call to update(), but applied recursively for all ancestors.

            :param leaf_value:      the value of subtree evaluation from the current player's perspective.

        """
        # If it is not root, this node's parent should be updated first.
        if self.PARENT:
            self.PARENT.update_recursive(-leaf_value)

        self.update(leaf_value)

    def get_value(self, c_puct: float) -> float:
        """ Calculate and return the value for this node: a combination of leaf evaluations, Q, and this node's prior
            adjusted for its visit count, u

            :param c_puct:      a number in (0, inf) controlling the relative impact of values, Q, and prior
                                probability, P, on this node's score.

        """
        self._u = c_puct * self.P * np.sqrt(self.PARENT.N_VISITS) / (1 + self.N_VISITS)

        return self.Q + self._u

    def __repr__(self) -> str:
        return "< TreeNode Parent {} with Children {} >".format(self.PARENT, self.CHILDREN)

    def __str__(self) -> str:
        return "< TreeNode Parent {} with Children {} >".format(self.PARENT, self.CHILDREN)
