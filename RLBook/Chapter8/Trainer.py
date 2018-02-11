# -*- coding: utf-8 -*-
""" RLBook.Chapter8.Trainer

1.  Run Episode
2.  Training
3.  Concatenate into Batch

"""
import logging
from copy import deepcopy

import numpy as np

from RLBook.Chapter8.Config import EnvConfig
from RLBook.Chapter8.MCTS import MonteCarloTreeSearch
from RLBook.Chapter8.TicTacToe import Game
from RLBook.Utils.Player import Player
from RLBook.Utils.Trainer import Trainer


class TicTacToeTrainer(Trainer):
    """ TicTacToe trainer
    """
    CHECKPOINT = 0

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

    def __repr__(self):
        return "< TicTacToe Trainer Class >"

    def __str__(self):
        return "< TicTacToe Trainer Class >"

    def run_episode(self, eval_phase=True):
        """

            :return:

        """
        new_game = deepcopy(self.ENV)
        winner = None

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
            move, action_prob = tree.recommended_play(eval_phase=True)
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

        # Return
        return 0 if winner is None else winner

    def self_play(self):
        """ Initialise the self playing process

            :return:

        """
        # Select the Training Agent
        player, player_index = self.GAME.nn_index
        other_player = self.GAME.competing_player
        best_fn = self.player_check(player=player)

        # Create a check-point
        player.model.save_checkpoint(filename=best_fn)
        player.check = 1

        for each_iteration in self.CONFIG.TRAINER.N_ITERATION:
            logging.info("Running training iteration: {}".format(each_iteration))

            for each_episode in self.CONFIG.TRAINER.EPISODE_MEM:
                logging.info("     Running training episode: {}".format(each_episode))

                # Run a Training episode
                _ = self.run_episode()

            # Running the training of NNet
            player.model.train(self.EPISODE_MEM)
            new_best_fn = self.player_check(player=player)

            # Once all the training has happened lets now update the underlying NNet Model
            player.model.save_checkpoint(filename=new_best_fn)
            player.check = 1

            # Allow the models to compete against one another
            win_ratio = self.dueling(nb_trials=self.CONFIG.N_DUELS, player_val=player.value)

            # If the Win ratio is greater than the Min Win Ratio then replace - otherwise revert
            if win_ratio > self.CONFIG.WIN_RATIO:
                # Updating the Competing Player
                logging.info("Replacing: {}".format(other_player.value))
                best_fn = new_best_fn

                # Replace the self play model
                other_player.model.load_checkpoint(filename=best_fn)
            else:
                logging.info("Reverting the Model to the previous version...")
                player.model.load_checkpoint(filename=best_fn)

    def dueling(self, nb_trials=None, player_val=0):
        """ Dueling the two models against one another

            :param nb_trials:       Number of Trials to Evaluate
            :param player_val:      Player Coin to check the Win Count against
            :return:

        """
        # Start the counting and prepare the meta-data
        nb_trials = self.CONFIG.trainer.EVALUATIONS if nb_trials is None else nb_trials
        n_wins, n_plays = 0, 0

        # Iterate from the number of trials
        for iteration in range(nb_trials):
            n_plays += 1

            # Run the episode and determine if
            if self.run_episode(eval_phase=False) == player_val:
                n_wins += 1

            # Record the stats
            logging.info("Dueling iteration: {} | Wins {} | NPlays {} |>".format(iteration, n_wins, n_plays))

        # Return
        return n_wins / n_plays if n_wins > 0 else 0

    def human_play(self, human, agent: Player):
        """ Human vs AI Agent

            :return:

        """
        raise NotImplementedError
