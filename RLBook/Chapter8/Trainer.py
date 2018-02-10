# -*- coding: utf-8 -*-
""" RLBook.Chapter8.Trainer

1.  Run Episode
2.  Training
3.  Concatenate into Batch

"""
from copy import deepcopy

import numpy as np

from RLBook.Chapter8.MCTS import MonteCarloTreeSearch
from RLBook.Chapter8.TicTacToe import Game
from RLBook.Utils.Trainer import Trainer


class TicTacToeTrainer(Trainer):
    """ TicTacToe trainer
    """

    def __init__(self, environment=Game(), trainer_config=None):
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
        new_game_states = []

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
            move = tree.recommended_play()
            new_game.play(move=move)
            new_game_states.append(move)

            # Show the tree and board, debugging
            tree.show_tree(level=1)
            new_game.show_board()

        if new_game.winner is not None:
            # Remove the first item - the first element in the list is just used for viz
            new_game.history.pop(0)

            # Get the NNet player - this will ensure that the first index in the states (b, p, n, n) p is the players
            winner = new_game.winner.value

            # Initialise the Arrays for Training
            batch_n = len(new_game.history)
            states, actions, score = np.zeros((batch_n, 2, 3, 3)), np.zeros((batch_n, 9)), np.zeros(
                (batch_n, 1)) + winner
            coin, index_player = new_game.get_nn_player_index

            for ind, (state, action, player) in enumerate(new_game.history):
                # Translate the States
                for index in range(2):
                    if index == new_game.get_nn_player_index:
                        states[ind, index, :, :] = np.abs(np.where(state == coin, state, 0))
                    else:
                        states[ind, index, :, :] = np.abs(np.where(state != coin, state, 0))
                # Get the action
                actions[ind, action] = 1

            self.EPISODE_MEM.append((states, actions, score))

    def self_play(self):
        """

            :return:

        """
        for each_iteration in self.CONFIG.TRAINER.N_ITERATION:
            print("Running training iteration: {}".format(each_iteration))

            for each_episode in self.CONFIG.TRAINER.EPISODE_MEM:
                print("     Running training episode: {}".format(each_episode))

                # Run a Training episode
                self.EPISODE_MEM()

            # TODO: Running the training of NNet
            # TODO: Once all the training has happened lets now update the underlying NNet Model
            # TODO: Allow the models to compete against one another
            raise NotImplementedError

    def human_play(self):
        """

            :return:

        """
        # TODO: Implement this...
        raise NotImplementedError

    def dueling(self, nb_trials=None, player_val=1):
        """

            :param nb_trials:       Number of Trials to Evaluate
            :param player_val:      Player Coin to check the Win Count against
            :return:

        """
        # nb_trials = self.CONFIG.trainer.EVALUATIONS if nb_trials is None else nb_trials
        # wins = 0

        # Iterate from the number of trials
        # for _ in range(nb_trials):
        # TODO: Implement this...
        raise NotImplementedError
