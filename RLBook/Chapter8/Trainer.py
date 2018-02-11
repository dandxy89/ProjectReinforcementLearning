# -*- coding: utf-8 -*-
""" RLBook.Chapter8.Trainer

1.  Run Episode
2.  Training
3.  Concatenate into Batch

"""
from copy import deepcopy

import numpy as np

from RLBook.Chapter8.Config import EnvConfig
from RLBook.Chapter8.MCTS import MonteCarloTreeSearch
from RLBook.Chapter8.TicTacToe import Game
from RLBook.Utils.Trainer import Trainer


class TicTacToeTrainer(Trainer):
    """ TicTacToe trainer
    """

    def __init__(self, environment=Game(), trainer_config=EnvConfig()):
        """ Initialise a Tic-Tac-Toe trainer

            Used for:

                *   Dueling (Against another fully trained AI...)
                *   Human play (Requires input...)
                *   Self Play (training a NNet)

            :param environment:     Game or Env
            :param trainer_config:  All configuration settings for the Trainer

        """
        super().__init__(environment=environment, trainer_config=trainer_config)

    def run_episode(self):
        """

            :return:

        """
        new_game = deepcopy(self.ENV)

        # Play until end
        while new_game.legal_plays():
            # Rollout the Tree
            tree = MonteCarloTreeSearch(game=new_game,
                                        evaluation_func=new_game.player.func,
                                        node_param=new_game.player.mcts_params,
                                        use_nn=new_game.player.use_nn)

            # Run the Tree Search
            tree.search(*new_game.player.mcts_search)

            # Play the recommended move and store the move
            move, action_prob = tree.recommended_play(train=True)
            new_game.play(move=move, action_prob=action_prob)

        if new_game.winner is not None:
            # Remove the first item - the first element in the list is just used for viz
            new_game.history.pop(0)

            # Get the NNet player - this will ensure that the first index in the states (b, p, n, n) p is the players
            winner = new_game.winner.value

            # Initialise the Arrays for Training
            coin, index_player = new_game.nn_index

            for ind, (state, action, player, move_prob) in enumerate(new_game.history):
                # Get empty templates
                states, actions, scores = np.zeros((1, 2, 3, 3)), np.zeros((1, 9)), np.zeros((1, 1)) + winner

                # Translate the States
                for index in range(2):
                    if index == index_player:
                        states[ind, index, :, :] = np.abs(np.where(state == coin.value, state, 0))
                    else:
                        states[ind, index, :, :] = np.abs(np.where(state != coin.value, state, 0))

                # Store the information and use later...
                self.EPISODE_MEM.append((states, move_prob, scores))

    def self_play(self):
        """

            :return:

        """
        for each_iteration in self.CONFIG.TRAINER.N_ITERATION:
            print("Running training iteration: {}".format(each_iteration))

            for each_episode in self.CONFIG.TRAINER.EPISODE_MEM:
                print("     Running training episode: {}".format(each_episode))

                # Run a Training episode
                self.run_episode()

            # Select the Training Agent
            player, player_index = self.GAME.nn_index

            # Running the training of NNet
            player.model.train(self.EPISODE_MEM)

            # Once all the training has happened lets now update the underlying NNet Model
            player.model.save_checkpoint(filename="TODO")

            # Allow the models to compete against one another
            win_ratio = self.dueling(nb_trials=self.CONFIG.N_DUELS)

            if win_ratio > self.CONFIG.WIN_RATIO:
                # Updating the Competing Player
                print("Replace: {}".format(player_index))

            raise NotImplementedError

    def human_play(self):
        """

            :return:

        """
        raise NotImplementedError

    def dueling(self, nb_trials=None, player_val=1):
        """

            :param nb_trials:       Number of Trials to Evaluate
            :param player_val:      Player Coin to check the Win Count against
            :return:

        """
        nb_trials = self.CONFIG.trainer.EVALUATIONS if nb_trials is None else nb_trials
        # wins = 0
        # Iterate from the number of trials
        # for _ in range(nb_trials):
        # TODO: Implement this...
        raise NotImplementedError
