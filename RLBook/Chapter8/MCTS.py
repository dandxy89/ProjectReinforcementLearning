# -*- coding: utf-8 -*-
""" RLBook.Chapter8.TicTacToeMCTS

*   Monte Carlo Tree Search implementation

"""
import logging
import time
from copy import deepcopy

import numpy as np
from anytree import Node, LevelOrderGroupIter, RenderTree

from RLBook.Chapter8 import DEFAULT_NODE_PARAMS
from RLBook.Utils.MathOperations import upper_confidence_bound


class MonteCarloTreeSearch:
    """ Implementation of the Monte Carlo Tree Search algorithm

        Note: Based on http://mcts.ai/pubs/mcts-survey-master.pdf

    """
    OUT = '%s | Action: %s | Player %s | %s Wins / %s Plays | V %.3f | Q: %.3f | U: %.3f | p: %.3f | Q+U %.3f |>'

    def __init__(self, game, evaluation_func, node_param=DEFAULT_NODE_PARAMS, use_nn=False):
        """ Initialise a Monte Carlo Tree Search

            :param game:                Board Game
            :param evaluation_func:     Evaluation function - Value, Policy function
            :param node_param:          Node parameters

        """
        self.GAME = game
        self.players = game.players
        self.policy = evaluation_func

        self.node_init_params = node_param
        self.root = Node('0', GAME=self.GAME, **self.node_init_params)

        self.use_nn = use_nn

    def selection(self, scoring_func=upper_confidence_bound, prob=0.12):
        """ Select a node of the tree based on scores or expand current one (if not all children have been visited)

            :param scoring_func:        the function that takes as inputs (n_plays, n_wins, n_ties) and output
                                        the node score
            :param prob:                Random factor to avoid over selecting the max all the time
            :return:                    Node with best score

        """
        # Selection should start from root node
        node = self.root

        # Browse each level until we reach a terminal node
        while node.children:
            # If node still has unexplored children we select it
            if len(node.GAME.legal_plays()) > len(node.children):
                return node

            # We go down the tree until we reach the bottom always choosing the best score at each level
            else:
                nodes = node.children
                scores = [scoring_func(node=node_) for node_ in nodes]

                # Select actions among children that gives maximum action value
                if np.random.rand(1) < prob:
                    node = nodes[np.random.randint(len(scores))]
                else:
                    node = nodes[np.argmax(scores)]

        return node

    def expansion(self, parent):
        """ Randomly expand a child for selected node in order to expand the tree

            :param parent:              Node to expand
            :return:                    Expanded child node

        """
        # Filter out plays that already have been expanded
        already_played = [node.GAME.last_play for node in parent.children]
        unexplored_plays = [play for play in parent.GAME.legal_plays() if play not in already_played]

        if unexplored_plays:
            # Choose one play randomly
            index = np.random.choice(len(unexplored_plays), 1)[0]
            selected_play = unexplored_plays[index]

            # Create a new node where this play is performed
            child_game = deepcopy(parent.GAME)
            child_game.play(selected_play)
            child_name = parent.name + '_' + str(len(parent.children))

            # Pass the Properties
            properties = deepcopy(self.node_init_params)
            if self.use_nn:
                # Get empty templates
                states, state, value = np.zeros((1, 2, 3, 3)), parent.GAME.player.value, parent.GAME.nn_index[0]

                # Translate the States
                for index in range(2):
                    if index == value:
                        states[0, index, :, :] = np.abs(np.where(state == value, state, 0))
                    else:
                        states[0, index, :, :] = np.abs(np.where(state != value, state, 0))

                # Evaluate the leaf using a network (value & policy) which outputs a list of (action, probability)
                # tuples p and also a score v in [-1, 1] for the current player.
                action_prob, leaf_value = self.policy(states)
                properties["PRIOR"] = action_prob[self.GAME.translate(selected_play)][1]
                properties["V"] = leaf_value
            else:
                properties["PRIOR"] = 1

            # Action picked
            properties["ACTION"] = parent.GAME.translate(selected_play)

            # Create the Child
            node = Node(name=child_name, parent=parent, GAME=child_game, **properties)

        # If all nodes have been explored return parent without expanding (can happen at end of tree search)
        else:
            node = parent

        return node

    def simulation(self, node):
        """ Simulate games from current game state and returns number of wins

            :param node:            Node from which the simulated games start
            :return:                Number of time the current player has won

        """
        # Play a game until the end
        game = deepcopy(node.GAME)

        while game.legal_plays():
            game.play()

        if game.winner == self.root.GAME.current_player:
            return 1 + np.random.rand() * 1e-6
        elif game.winner is None:
            return 0
        else:
            return -1 - np.random.rand() * 1e-6

    def backpropagate(self, node, leaf_value):
        """ Back-propagate the results of the simulations to the ancestor nodes of the tree

            :param node:        Starting node for backpropagation (from bottom to top)
            :param leaf_value:  Leaf node value

        """
        # Apply updates on current node
        node.N_PLAYS += 1
        node.N_WINS += 1 if leaf_value >= 1 else 0
        node.N_TIES += 1 if leaf_value == 0 else 0
        node.Q = self.update_q(leaf_value=leaf_value, node=node)

        # All all of the ancestors
        for ancestor in node.ancestors:
            ancestor.N_PLAYS += 1
            ancestor.N_WINS += 1 if leaf_value >= 1 else 0
            ancestor.N_TIES += 1 if leaf_value == 0 else 0
            ancestor.Q = self.update_q(leaf_value=leaf_value, node=ancestor)

    @staticmethod
    def update_q(leaf_value, node):
        """ Update function for Q

            :param leaf_value:      Leaf value
            :param node:            Node from in the MCTS tree
            :return:                Calculated Q value

        """
        return leaf_value if node.Q == 0 else (node.N_PLAYS * node.Q + leaf_value) / (node.N_PLAYS + 1)

    @staticmethod
    def sort_by_move(nodes):
        """ Sort nodes by move from (0, 0) to (2,2)

            :param nodes:       List of nodes
            :return:            Sorted list

        """
        return sorted(nodes, key=lambda n: n.GAME.last_play)

    def show_tree(self, level=-1):
        """ Print the current state of the tree along with some statistics on nodes

            :param level:               Max level to print. If -1 print full tree
            :return:                    Tree representation as a string or nothing if printed

        """
        result = []

        # From list of tuples of nodes to list of nodes
        if level > 0:
            nodes_selections = [e for sub in list(LevelOrderGroupIter(self.root))[:level + 1] for e in sub]
        else:
            nodes_selections = []

        # Iterate through the Tree and construct the Output
        for indent, _, node in RenderTree(self.root, childiter=self.sort_by_move):
            if level == -1 or node in nodes_selections:
                result.append((self.OUT % (indent, node.ACTION, node.GAME.current_player.display, node.N_WINS,
                                           node.N_PLAYS, node.V, node.Q, node.U,
                                           node.PRIOR, node.Q + node.U)))

        # Display the result
        print('\n'.join(result))

    def search(self, max_iterations=5000, max_runtime=20):
        """ Run a Monte Carlo Tree Search starting from root node

            Defaults:
                - max_iterations      20000
                - max_runtime         20

            :param max_iterations:      max number of iterations for the tree search
            :param max_runtime:         max search time in seconds

        """
        t1 = time.time()

        # Iterate for the maximum number of iterations
        for _ in range(max_iterations):
            # Selection
            node = self.selection()

            # Expansion
            expanded_node = self.expansion(parent=node)

            # Simulation - play out the game to Termination
            leaf_value = self.simulation(node=expanded_node)

            # Back propagate the result
            self.backpropagate(node=expanded_node, leaf_value=leaf_value)

            # Early exit if and only if the time taken to solve > max_runtime
            if time.time() - t1 > max_runtime:
                logging.warning("TimeOut during the searching phase.")
                break

    def recommended_play(self, train=True):
        """ Move recommended by the Monte Carlo Tree Search

            :return:        A tuple corresponding to the recommended move

        """
        nodes = list(LevelOrderGroupIter(self.root))

        if nodes:
            action_prob = np.zeros((1, 9))
            for n in nodes[1]:
                action_prob[0, n.ACTION] = n.PRIOR

            if train:
                logging.debug("Using a stochastic action selection")
                return self.stochastic_action(nodes[1]).GAME.last_play, action_prob
            else:
                logging.debug("Using the U + Q strategy used in AlphaZero")
                return self.deterministic_action(nodes[1]).GAME.last_play, action_prob

    @staticmethod
    def stochastic_action(nodes):
        """ Non-uniform Action selection

            :param nodes:       List of Nodes with their respective node properties
            :return:            'Best Node' class

        """
        records = np.array([np.power(n.U, 1 / n.TAU) for val, n in enumerate(nodes)])
        records /= records.sum()

        return nodes[np.random.choice(len(records), p=records)]

    @staticmethod
    def deterministic_action(nodes):
        """

            :param nodes:       List of Nodes with their respective node properties
            :return:            'Best Node' class

        """
        length = len(nodes)
        records = np.array([(n.U + n.Q + 1000) * length for val, n in enumerate(nodes)])
        records /= records.sum()

        return nodes[np.argmax(records)]
