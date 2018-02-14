# -*- coding: utf-8 -*-
""" RLBook.Chapter8.Trainer

1.  Run Episode
2.  Training
3.  Concatenate into Batch

"""
import logging
import pickle
from copy import deepcopy

import numpy as np

from RLBook.Chapter8.Config import EnvConfig
from RLBook.Chapter8.MCTS import MonteCarloTreeSearch
from RLBook.Chapter8.NNetPlayers import create_keras_models
from RLBook.Chapter8.TicTacToe import Game
from RLBook.Utils.Player import Player
from RLBook.Utils.Trainer import Trainer


class TicTacToeTrainer(Trainer):
    """ TicTacToe trainer
    """
    CHECKPOINT = 0

    def __init__(self, environment=Game(), trainer_config={}, eval_functions=create_keras_models(),
                 memory: list = None):
        """ Initialise a Tic-Tac-Toe trainer

            Used for:

                *   Dueling (Against another fully trained AI...)
                *   Human play (Requires input...)
                *   Self Play (training a NNet)

            :param environment:     Game or Env
            :param trainer_config:  All configuration settings for the Trainer
            :param eval_functions:  One evaluation function per player
            :param memory:          List of the all the Memories captured during training

        """
        super().__init__(environment=environment, trainer_config=EnvConfig(**trainer_config))
        self.eval_function = eval_functions

        if memory is not None:
            self.EPISODE_MEM = memory

    def __repr__(self):
        return "< TicTacToe Trainer Class >"

    def __str__(self):
        return "< TicTacToe Trainer Class >"

    def run_episode(self, eval_phase=True):
        """ This function will run only one episode until the game terminates

            If there is a winner the game states will be added to the Memory

        """
        new_game = deepcopy(self.GAME)
        winner = None

        # Alternate Starting Player
        if np.random.random() > 0.5:
            new_game.current_player = next(new_game.players_gen)

        # Play until end
        while new_game.legal_plays():
            # Rollout the Tree
            tree = MonteCarloTreeSearch(game=new_game,
                                        # Any model should have a predict method passed
                                        evaluation_func=self.eval_function[str(new_game.player.value)].predict,
                                        node_param=new_game.player.mcts_params,
                                        use_nn=new_game.player.use_nn)

            # Run the Tree Search
            tree.search(*new_game.player.mcts_search)

            # Play the recommended move and store the move
            move, action_prob = tree.recommended_play(train=eval_phase)
            tree.show_tree(level=1)
            new_game.play(move=move, action_prob=action_prob)

            logging.info("Showing board!")
            new_game.show_board()

        if new_game.winner is not None:
            logging.info("Winner found: {}".format(new_game.winner))
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
                        states[0, index, :, :] = np.abs(np.where(state == coin.value, state, 0))
                    else:
                        states[0, index, :, :] = np.abs(np.where(state != coin.value, state, 0))

                # Store the information and use later...
                self.EPISODE_MEM[0].append(states)
                self.EPISODE_MEM[1].append(move_prob)
                self.EPISODE_MEM[2].append(scores)

        # Return
        return 0 if winner is None else winner

    def self_play(self):
        """ Initialise the self playing process

            :return:        Training by self play

        """
        # Select the Training Agent
        player, player_index = self.GAME.nn_index
        other_player = self.GAME.competing_player
        best_fn = self.player_check(player=player)

        # Create a check-point
        model = self.eval_function[str(player.value)]
        model.save_checkpoint(filename=best_fn)
        player.check = 1

        # Warm up Iterations to populate the Memory
        for each_warm_up in range(self.CONFIG.WARM_UP_ITERATION):
            logging.info("     Running warm up episode: {}".format(each_warm_up + 1))

            # Run a Training episode
            self.run_episode()

        for each_iteration in range(self.CONFIG.N_ITERATION):
            logging.info("Running training iteration: {}".format(each_iteration + 1))

            for each_episode in range(self.CONFIG.N_EPISODE):
                logging.info("     Running training episode: {}".format(each_episode + 1))

                # Run a Training episode
                self.run_episode()

                # Running the training of NNet
                model.train(self.EPISODE_MEM)

            # Pickle the Memory
            self.pickle_memory()

            # Get the Best Function
            new_best_fn = self.player_check(player=player)

            # Once all the training has happened lets now update the underlying NNet Model
            model.save_checkpoint(filename=new_best_fn)
            player.check = 1

            # Allow the models to compete against one another
            win_ratio = self.dueling(nb_trials=self.CONFIG.N_DUELS, player_val=player.value)

            # If the Win ratio is greater than the Min Win Ratio then replace - otherwise revert
            if win_ratio > self.CONFIG.WIN_RATIO:
                # Updating the Competing Player
                logging.info("Replacing: {}".format(other_player.value))
                best_fn = new_best_fn

                # Replace the self play model
                other_model = self.eval_function[str(other_player.value)]
                other_model.load_checkpoint(filename=best_fn)
            else:
                logging.info("Reverting the Model to the previous version...")
                model.load_checkpoint(filename=best_fn)

            # Pickle the Memory
            self.pickle_memory()

    def pickle_memory(self):
        """ Store all the Memory for use later
        """
        logging.info("Pickling Memories.")
        with open("memory.pickle", "wb") as handle:
            pickle.dump(self.EPISODE_MEM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def dueling(self, nb_trials=None, player_val=0):
        """ Dueling the two models against one another

            :param nb_trials:       Number of Trials to Evaluate
            :param player_val:      Player Coin to check the Win Count against
            :return:                Win Ratio

        """
        # Start the counting and prepare the meta-data
        nb_trials = self.CONFIG.EVALUATIONS if nb_trials is None else nb_trials
        n_wins, n_plays = 0, 0

        # Iterate from the number of trials
        for iteration in range(nb_trials):
            n_plays += 1

            # Run the episode and determine if it classes as a win... including draws
            winner = self.run_episode(eval_phase=False)
            if winner == player_val or winner == 0:
                n_wins += 1

            # Record the stats
            logging.info("Dueling iteration: {} | Wins {} | NPlays {} |>".format(iteration, n_wins, n_plays))

        # Return
        return n_wins / n_plays if n_wins > 0 else 0

    def human_play(self, human, agent: Player):
        """ Human vs AI Agent

            TODO: Implement this!

            :return:

        """
        raise NotImplementedError
