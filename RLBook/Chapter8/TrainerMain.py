# -*- coding: utf-8 -*-
""" RLBook.Chapter8.TrainerMain
"""
import logging

from RLBook.Chapter8.NNetPlayers import NNetPlayers
from RLBook.Chapter8.TicTacToe import Game
from RLBook.Chapter8.Trainer import TicTacToeTrainer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")


def train_model():
    # params:
    trainer_config = {}
    game = Game(players=NNetPlayers, using_nn=True, nn_player=0)

    # Initialise a TicTacToe Trainer
    trainer = TicTacToeTrainer(environment=game, trainer_config=trainer_config)

    # Commence self-play
    trainer.self_play()


if __name__ == '__main__':
    train_model()
