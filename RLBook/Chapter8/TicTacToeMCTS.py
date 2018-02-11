# -*- coding: utf-8 -*-
""" Chapter8.MCTS

-   Simple implementation demonstrating using MCTS for the purpose of playing the game of TicTacToe

"""
from copy import deepcopy

from RLBook.Chapter8.MCTS import MonteCarloTreeSearch
from RLBook.Chapter8.TicTacToe import Game, DEFAULT_PLAYERS


def main():
    """ Play two Monte Carlo Tree searches against one another...
    """
    # params:
    g = Game(players=DEFAULT_PLAYERS)

    new_game = deepcopy(g)

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
        new_game.play(move=move)

        # Show the tree and board, debugging
        tree.show_tree(level=1)
        new_game.show_board()


if __name__ == '__main__':
    main()
