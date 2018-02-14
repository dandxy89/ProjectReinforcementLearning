# -*- coding: utf-8 -*-
""" RLBook.Chapter8.TrainerMain

-   The purpose of this function is to train two self playing agents against one another in similar
    fashion as to how AlphaZero was reported to have played.

"""
import logging
import pickle

from RLBook.Chapter8.Config import Config
from RLBook.Chapter8.NNetPlayers import NNetPlayers, create_keras_models
from RLBook.Chapter8.TicTacToe import Game
from RLBook.Chapter8.Trainer import TicTacToeTrainer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")


def train_model():
    # custom params
    trainer_config = {}
    nn_net_one = {}
    nn_net_two = {}
    with open("ttt_memory.pickle", "rb") as handle:
        memory_data = pickle.load(handle)

    # Initialise both the Game and each Players Models
    game = Game(players=NNetPlayers, using_nn=True, nn_player=0)
    models = create_keras_models(config1=Config(**nn_net_one),
                                 config2=Config(**nn_net_two))

    # Initialise a TicTacToe Trainer
    trainer = TicTacToeTrainer(environment=game,
                               trainer_config=trainer_config,
                               eval_functions=models,
                               memory=memory_data)

    # Commence self-play
    trainer.self_play()


if __name__ == "__main__":
    train_model()
