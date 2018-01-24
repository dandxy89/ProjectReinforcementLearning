# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" Chapter1.Environment
"""
import logging
import random
from collections import defaultdict

import numpy as np

from RLBook.Chapter1.TicTacToe import TicTacToeGame

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("Chapter1")


class GameState:
    """ Game End States
    """
    WIN = "Win!"
    LOSS = "Loss!"
    DRAW = "Draw!"
    IN_PROGRESS = "Still playing..."

    def __repr__(self):
        return "< Tic Tac Toe Game States >"


class TicTacToeEnvironment:
    """ Tic-Tac-Toe Modelling Environment
    """
    AGENT1 = None
    AGENT2 = None
    DEFAULT_BOARD = None
    N_TRIALS = None
    SCORES = defaultdict(int)

    def __init__(self, agent_one, agent_two, trials: int = 100, board_size=(3, 3), switch=None):
        """ Initialise a Modelling Environment

            :param agent_one:       Agent - either an AI or a Human player
            :param agent_two:       Agent - either an AI or a Human player
            :param trials:          Number of Games
            :param board_size:      Tic-Tac-Toe Game

        """
        self.AGENT1 = agent_one
        self.AGENT2 = agent_two

        self.DEFAULT_BOARD = board_size
        self.N_TRIALS = trials

        if switch is not None:
            self.SWITCH_PROB = switch

    def __repr__(self):
        return "< Tic-Tac-Toe Modelling Environment >"

    def __str__(self):
        return "< Tic-Tac-Toe Modelling Environment >"

    def train(self):
        from tqdm import tqdm
        # For each Trial play a Game of Tic-Tac-Toe
        for each_trial in tqdm(range(self.N_TRIALS)):
            # Play a Game until termination
            self.play_game()

    def switch_role(self, id: str):
        """ Switch which Agent starts the Game

            :param id:
            :return:

        """
        return self.AGENT2 if id == self.AGENT1.ID else self.AGENT1

    @staticmethod
    def get_eligible_indexes(array: np.ndarray) -> np.ndarray:
        return np.where(array.flatten() == 0)

    def update_reward(self, game, action):
        # Agent 1 Wins!
        if game.WINNER == 1:
            logger.debug("Agent 1 Wins!")
            self.AGENT1.append(state=game.get_state(), reward=game.REWARD_WIN, action=action)
            self.AGENT1.update_policy()

            self.AGENT2.append(state=game.get_state(), reward=game.REWARD_LOSS, action=action)
            self.AGENT2.update_policy()

        # Agent 2 Wins!
        elif game.WINNER == -1:
            logger.debug("Agent 2 Wins!")
            self.AGENT2.append(state=game.get_state(), reward=game.REWARD_WIN, action=action)
            self.AGENT2.update_policy()

            self.AGENT1.append(state=game.get_state(), reward=game.REWARD_LOSS, action=action)
            self.AGENT1.update_policy()

        # Agent 1 and Agent 2 draw!
        else:
            logger.debug("Agent 1 and Agent 2 draw!")
            self.AGENT2.append(state=game.get_state(), reward=game.REWARD_DRAW, action=action)
            self.AGENT2.update_policy()

            self.AGENT1.append(state=game.get_state(), reward=game.REWARD_DRAW, action=action)
            self.AGENT1.update_policy()

        # Reset the Agent Storage
        self.AGENT1.reset()
        self.AGENT2.reset()

    def play_game(self):
        in_progress = True
        active_gamer = self.AGENT1 if random.random() < 0.5 else self.AGENT2
        game = TicTacToeGame(columns=self.DEFAULT_BOARD[1], rows=self.DEFAULT_BOARD[0])

        # While there are available states continue playing!
        while in_progress:
            # Get the current State
            game_state = game.get_state()

            # Get the eligible Actions
            eligible_actions = self.get_eligible_indexes(array=game.BOARD)

            # Let the Agent take an action based on the state
            action = active_gamer.take_action(game_state, eligible_actions[0])
            active_gamer.append(state=game_state, action=action, reward=game.REWARD_IN_PROGRESS)

            # Update the Board with the Desired Action
            game.update_board(action=action, value=active_gamer.ACTION)

            # Check if the Game is complete...
            if game.is_end_state():
                self.update_reward(game=game, action=action)
                in_progress = False
                self.SCORES[game.WINNER] += 1

            # Switch to the Other Player
            active_gamer = self.switch_role(id=active_gamer.ID)
