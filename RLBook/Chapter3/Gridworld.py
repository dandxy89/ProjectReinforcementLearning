# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.Chapter3.GridWorld

"""
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_ACTION_PROB = {'L': 0.25, 'U': 0.25, 'R': 0.25, 'D': 0.25}


class GridWorld:
    """ Grid World from Chapter 3
    """

    WORLD_SIZE = 5
    A_POS = [0, 1]
    A_PRIME_POS = [4, 1]
    B_POS = [0, 3]
    B_PRIME_POS = [2, 3]
    DISCOUNT_RATE = 0.9

    SET_ACTION_PROB = False

    WORLD = np.zeros((WORLD_SIZE, WORLD_SIZE))

    # left, up, right, down
    ACTIONS = ['L', 'U', 'R', 'D']
    ACTION_PROBABILITIES = []

    NEXT_STATE = []
    ACTION_REWARD = []

    TOL = 1e-4
    ACCEPTED_ARGUMENTS = ["TOL", "DISCOUNT_RATE"]

    def __init__(self, **kwargs):
        # Set the Attributes
        for k in kwargs.keys():
            if k in self.ACCEPTED_ARGUMENTS:
                self.__setattr__(k, kwargs[k])

    def set_action_probabilities(self, probs=DEFAULT_ACTION_PROB):
        """ Set the Action Probabilities

            :return:

        """
        for i in range(0, self.WORLD_SIZE):
            self.ACTION_PROBABILITIES.append([])
            for j in range(0, self.WORLD_SIZE):
                self.ACTION_PROBABILITIES[i].append(probs)

        self.SET_ACTION_PROB = True

    def find_possible_states(self):
        """

            :return:

        """
        if not self.SET_ACTION_PROB:
            self.set_action_probabilities()

        for i in range(0, self.WORLD_SIZE):
            self.NEXT_STATE.append([])
            self.ACTION_REWARD.append([])

            for j in range(0, self.WORLD_SIZE):
                next = dict()
                reward = dict()

                if i == 0:
                    next['U'] = [i, j]
                    reward['U'] = -1.0
                else:
                    next['U'] = [i - 1, j]
                    reward['U'] = 0.0

                if i == self.WORLD_SIZE - 1:
                    next['D'] = [i, j]
                    reward['D'] = -1.0
                else:
                    next['D'] = [i + 1, j]
                    reward['D'] = 0.0

                if j == 0:
                    next['L'] = [i, j]
                    reward['L'] = -1.0
                else:
                    next['L'] = [i, j - 1]
                    reward['L'] = 0.0

                if j == self.WORLD_SIZE - 1:
                    next['R'] = [i, j]
                    reward['R'] = -1.0
                else:
                    next['R'] = [i, j + 1]
                    reward['R'] = 0.0

                if [i, j] == self.A_POS:
                    next['L'] = next['R'] = next['D'] = next['U'] = self.A_PRIME_POS
                    reward['L'] = reward['R'] = reward['D'] = reward['U'] = 10.0

                if [i, j] == self.B_POS:
                    next['L'] = next['R'] = next['D'] = next['U'] = self.B_PRIME_POS
                    reward['L'] = reward['R'] = reward['D'] = reward['U'] = 5.0

                self.NEXT_STATE[i].append(next)
                self.ACTION_REWARD[i].append(reward)

    def run_bellman(self):
        """ Run the Bellman Iteration method
        
            :return:    Value state function
             
        """
        world = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))

        # Run until Convergence has been achieve to a given tolerance
        while True:
            new_world = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))

            for i in range(0, self.WORLD_SIZE):
                for j in range(0, self.WORLD_SIZE):
                    for action in self.ACTIONS:
                        new_position = self.NEXT_STATE[i][j][action]

                        # Bellman equation
                        new_world[i, j] += self.ACTION_PROBABILITIES[i][j][action] * \
                                           (self.ACTION_REWARD[i][j][action] +
                                            self.DISCOUNT_RATE * world[new_position[0], new_position[1]]
                                            )

            if np.sum(np.abs(world - new_world)) < self.TOL:
                break
            else:
                world = new_world

        return world

    def run_value_iteration(self) -> np.ndarray:
        """ Run the Value Iteration method

            :return:    Value state function

        """
        world = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))

        # Run until Convergence has been achieve to a given tolerance
        while True:

            new_world = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))
            for i in range(0, self.WORLD_SIZE):
                for j in range(0, self.WORLD_SIZE):
                    values = []
                    for action in self.ACTIONS:
                        new_position = self.NEXT_STATE[i][j][action]

                        # Value iteration
                        values.append(self.ACTION_REWARD[i][j][action] +
                                      self.DISCOUNT_RATE * world[new_position[0], new_position[1]]
                                      )

                    new_world[i][j] = np.max(values)

            if np.sum(np.abs(world - new_world)) < self.TOL:
                break
            else:
                world = new_world

        return world

    @staticmethod
    def plot_mesh(world: np.ndarray, title: str, file_name: str):
        """ Plot the Array as a Mesh with Matplotlib

            :param world:       Numpy Array
            :param title:       Chart title
            :param file_name:        Name of the file

        """
        plt.figure()
        plt.pcolor(world)
        plt.colorbar()
        plt.title("{}".format(title))
        plt.savefig("Plots/Chapter3/{}.png".format(type))


if __name__ == '__main__':
    # Initialise the Grid World
    env = GridWorld(**{"TOL": 1e-8})

    # Find the next possible states and rewards
    env.find_possible_states()

    # Run the Bellman Version
    bellman_world = env.run_bellman()

    # Run the Value iteration version
    value_iteration = env.run_value_iteration()

    print("Bellman World...")
    pprint(bellman_world)
    print("Value Iteration World...")
    pprint(value_iteration)

    # Save the Plots for review in the future...
    env.plot_mesh(world=bellman_world, title="Bellman World", file_name="Bellman")
    env.plot_mesh(world=value_iteration, title="Value Iteration World", file_name="ValueIteration")
