# -*- coding: utf-8 -*-
""" Chapter5.MCTS

-   MCTS - Monte Carlo Tree Search

"""
import copy

import numpy as np

from RLBook.Chapter8.TreeNode import TreeNode
from RLBook.Utils.MathOps import softmax


class MCTS(object):
    """ An implementation of Monte Carlo Tree Search
    """

    def __init__(self, policy_value_fn, c_puct: float = 2, n_playout: int = 1000):
        """ Monte Carlo Tree Search initialisation

            :param policy_value_fn:     a function that takes in a board state and outputs a list of (action, probability)
                                        tuples and also a score in [-1, 1] (i.e. the expected value of the end game
                                        score from the current player's perspective) for the current player.
            :param c_puct:              a number in (0, inf) that controls how quickly exploration converges to the
                                        maximum-value policy, where a higher value means relying on the prior more
        """
        self.ROOT = TreeNode(None, 1.0)
        self.POLICY = policy_value_fn
        self.C_PUCT = c_puct
        self.N_PLAYOUT = n_playout

    def _playout(self, state):
        """ Run a single playout from the root to the leaf, getting a value at the leaf and
            propagating it back through its parents. State is modified in-place, so a copy must be provided.

            :param state:       A copy of the state.

        """
        node = self.ROOT
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self.C_PUCT)
            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of (action, probability) tuples p and also
        # a score v in [-1, 1] for the current player.
        action_probs, leaf_value = self.POLICY(state)

        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)

        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp: float = 1e-3):
        """ Runs all playouts sequentially and returns the available actions and their corresponding probabilities

            :param state:       the current state, including both game state and the current player.
            :param temp:        temperature parameter in (0, 1] that controls the level of exploration
            :return:

        """
        for n in range(self.N_PLAYOUT):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on the visit counts at the root node
        act_visits = [(act, node._n_visits) for act, node in self.ROOT.CHILDREN.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move: int):
        """ Step forward in the tree, keeping everything we already know about the subtree.

            :param last_move:
            :return:

        """
        if last_move in self.ROOT.CHILDREN:
            self.ROOT = self.ROOT.CHILDREN[last_move]
            self.ROOT.PARENT = None
        else:
            self.ROOT = TreeNode(None, 1.0)

    def __str__(self):
        return "< MCTS >"

    def __repr__(self):
        return "< MCTS >"


class MCTSInterface(object):
    """ AI Agent based on MCTS
    """
    MCTS = None
    PLAYER = None

    def __init__(self, policy_value_function, c_puct: float = 2, n_playout: int = 1000, is_selfplay: int = 0):
        """ Class to Action the Rollouts

            :param policy_value_function:
            :param c_puct:
            :param n_playout:
            :param is_selfplay:

        """
        self.MCTS = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_identity(self, p: int):
        self.PLAYER = p

    def reset_player(self):
        self.MCTS.update_with_move(-1)

    def get_action(self, game, temp: float = 1e-3, return_prob: float = 0):
        """

            :param game:
            :param temp:
            :param return_prob:
            :return:

        """
        available_moves = game.availables
        move_probabilities = np.zeros(len(available_moves))

        if len(available_moves) > 0:
            acts, probs = self.MCTS.get_move_probs(game, temp)
            move_probabilities[list(acts)] = probs

            if self._is_selfplay:
                # Add Dirichlet Noise for exploration (needed for self-play training)
                move = np.random.choice(acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))

                # Update the root node and reuse the search tree
                self.MCTS.update_with_move(move)
            else:
                # With the default temp=1e-3, this is almost equivalent to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)

                # Reset the root node
                self.MCTS.update_with_move(-1)

            if return_prob:
                return move, move_probabilities

            else:
                return move

        else:
            print("No available moves...")

    def __str__(self):
        return "< MCTS Rollout Class {} >".format(self.PLAYER)

    def __repr__(self):
        return "< MCTS Rollout Class {} >".format(self.PLAYER)
