# -*- coding: utf-8 -*-
""" RLBook.Chapter8.TicTacToeMCTS

*   Monte Carlo Tree Search implementation

"""
import time
from copy import deepcopy

import numpy as np
from anytree import Node, LevelOrderGroupIter, RenderTree

from RLBook.Chapter8 import DEFAULT_NODE_PARAMS
from RLBook.Utils.Decorators import timeit
from RLBook.Utils.MathOps import upper_confidence_bound


class MonteCarloTreeSearch:
    """ Implementation of the Monte Carlo Tree Search algorithm

        Note: Based on http://mcts.ai/pubs/mcts-survey-master.pdf

    """

    def __init__(self, game, evaluation_func, node_param=DEFAULT_NODE_PARAMS):
        """ Initialise a Monte Carlo Tree Search

            :param game:                Board Game
            :param evaluation_func:     Evaluation function - Value, Policy function
            :param node_param:          Node parameters

        """
        self.GAME = game
        self.policy = evaluation_func

        self.node_init_params = node_param
        self.root = Node('0', GAME=self.GAME, **self.node_init_params)

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
                scores = [scoring_func(total_plays=node_.parent.N_PLAYS, node=node_) for node_ in nodes]

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

        # Evaluate the leaf using a network (value & policy) which outputs a list of (action, probability)
        # tuples p and also a score v in [-1, 1] for the current player.
        action_probs, leaf_value = self.policy(parent.GAME.state)

        if unexplored_plays:
            # Choose one play randomly
            selected_play = unexplored_plays[np.random.choice(len(unexplored_plays), 1)[0]]

            # Create a new node where this play is performed
            child_game = deepcopy(parent.GAME)
            child_game.play(selected_play)
            child_name = parent.name + '_' + str(len(parent.children))

            # Pass the Properties
            properties = deepcopy(self.node_init_params)
            properties["PRIOR"] = 1  # action_probs[parent.game.translate(selected_play)][1]
            properties["action"] = parent.GAME.translate(selected_play)

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

        if game.winner() == self.root.GAME.current_player:
            return 1 + np.random.rand() * 1e-6
        elif game.winner() is None:
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
        node.Q = leaf_value if node.Q == 0 else (node.N_PLAYS * node.Q + leaf_value) / (node.N_PLAYS + 1)

        # All all of the ancestors
        for ancestor in node.ancestors:
            ancestor.N_PLAYS += 1
            ancestor.N_WINS += 1 if leaf_value >= 1 else 0
            ancestor.N_TIES += 1 if leaf_value == 0 else 0
            ancestor.Q = leaf_value if ancestor.Q == 0 else (ancestor.N_PLAYS *
                                                             ancestor.Q + leaf_value) / (ancestor.N_PLAYS + 1)

    @staticmethod
    def sort_by_move(nodes):
        """ Sort nodes by move from (0, 0) to (2,2)

            :param nodes:       list of nodes
            :return:            sorted list

        """
        return sorted(nodes, key=lambda n: n.GAME.last_play)

    def show_tree(self, level=-1):
        """ Print the current state of the tree along with some statistics on nodes

            :param level:               max level to print. If -1 print full tree
            :return:                    tree representation as a string or nothing if printed

        """
        result = ['\n']
        output = '%s | Action: %s | Player %s | %s Wins / %s Plays / WRatio %s | Q: %.3f | U: %.3f |>'
        nodes_selections = []

        # From list of tuples of nodes to list of nodes
        if level > 0:
            nodes_selections = [e for sub in list(LevelOrderGroupIter(self.root))[:level + 1] for e in sub]

        # Iterate through the Tree and construct the Output
        for indent, _, node in RenderTree(self.root, childiter=self.sort_by_move):
            if level == -1 or node in nodes_selections:
                result.append((output % (indent, node.action, node.GAME.current_player.display,
                                         node.N_WINS, node.N_PLAYS, node.N_WINS / node.N_PLAYS, node.Q, node.U)))

        # Display the result
        print('\n'.join(result))

    @timeit
    def search(self, max_iterations, max_runtime):
        """ Run a Monte Carlo Tree Search starting from root node

            :param max_iterations:      max number of iterations for the tree search
            :param max_runtime:         max search time in seconds

        """
        t1 = time.time()
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
                print("TimeOut!")
                break

    def recommended_play(self):
        """ Move recommended by the Monte Carlo Tree Search

            :return:        A tuple corresponding to the recommended move

        """
        nodes = list(LevelOrderGroupIter(self.root))
        if nodes:
            return self.another_action(nodes[1]).GAME.last_play

    @staticmethod
    def non_uniform_action(nodes):
        """ Non-uniform Action selection

            :param nodes:       List of Nodes with their respective node properties
            :return:            'Best Node' class

        """
        records = np.array([np.power(n.U, 1 / n.TAU) for val, n in enumerate(nodes)])
        records /= records.sum()
        return nodes[np.random.choice(len(records), p=records)]

    @staticmethod
    def another_action(nodes):
        """

            :param nodes:       List of Nodes with their respective node properties
            :return:            'Best Node' class

        """
        l = len(nodes)
        records = np.array([(n.U + n.Q + 1000) * l for val, n in enumerate(nodes)])
        records /= records.sum()
        return nodes[np.argmax(records)]
