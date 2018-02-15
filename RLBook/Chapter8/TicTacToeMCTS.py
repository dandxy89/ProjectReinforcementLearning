# -*- coding: utf-8 -*-
""" Chapter8.MCTS

-   Simple implementation demonstrating using MCTS for the purpose of playing the game of TicTacToe
-   Was used to ensure that the MCTS and Game were working as expecting whilst un-going changes

"""

from RLBook.Chapter8.Config import Config
from RLBook.Chapter8.DefaultPlayers import DEFAULT_PLAYERS
from RLBook.Chapter8.MCTS import MonteCarloTreeSearch
from RLBook.Chapter8.NNetPlayers import create_keras_models, NNetPlayers
from RLBook.Chapter8.TicTacToe import Game


def main():
    """ Play two Monte Carlo Tree searches against one another...
    """
    # ##########################################################
    # params (Testing w/o NN):
    new_game = Game(players=DEFAULT_PLAYERS)

    # Play until end
    while new_game.legal_plays() and new_game.winner is None:
        print(new_game.player)
        # Rollout the Tree
        tree = MonteCarloTreeSearch(game=new_game,
                                    evaluation_func=new_game.player.func,
                                    node_param=new_game.player.mcts_params,
                                    use_nn=new_game.player.use_nn)

        # Run the Tree Search
        tree.search(*new_game.player.mcts_search)

        # Find the recommended move
        # Note: Replace to use the Stochastic action selection
        move, _ = tree.recommended_play(train=False)
        # Stochastic: move, _ = tree.recommended_play(train=True)

        # Play the recommended move and store the move
        tree.show_tree(level=1)
        new_game.play(move=move)

        # Show the tree and board, debugging
        new_game.show_board()

    # ##########################################################
    # params (Testing w NN):
    game = Game(players=NNetPlayers, using_nn=True, nn_player=0)
    models = create_keras_models(config1=Config(**{}),
                                 config2=Config(**{}))
    models['1'].load_checkpoint("RLBook/Chapter8/20180214_KerasModel_TTT_V1")

    # Play until end
    while game.legal_plays() and game.winner is None:
        print(game.player)
        # Rollout the Tree
        tree = MonteCarloTreeSearch(game=game,
                                    evaluation_func=models[str(game.player.value)].predict,
                                    node_param=game.player.mcts_params,
                                    use_nn=game.player.use_nn)

        # Run the Tree Search
        tree.search(*game.player.mcts_search)

        # Find the recommended move
        # Note: Replace to use the Stochastic action selection
        move, _ = tree.recommended_play(train=False)
        # Stochastic: move, _ = tree.recommended_play(train=True)

        # Play the recommended move and store the move
        tree.show_tree(level=1)
        game.play(move=move)

        # Show the tree and board, debugging
        game.show_board()
    # ##########################################################


if __name__ == '__main__':
    main()
