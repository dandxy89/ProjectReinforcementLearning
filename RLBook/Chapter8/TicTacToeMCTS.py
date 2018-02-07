# -*- coding: utf-8 -*-
""" Chapter8.MCTS

-   Simple implementation demonstrating using MCTS for the purpose of playing the game of TicTacToe

"""
from RLBook.Chapter8.Config import Config
from RLBook.Chapter8.MCTS import MonteCarloTreeSearch
from RLBook.Chapter8.TicTacToe import Game
from RLBook.Utils.MathOps import random_value_policy

NODE_PARAMS = dict(X=dict(N_PLAYS=0, N_WINS=0, N_TIES=0, SCORE=0., PRIOR=0,
                          C_PUCT=6., TAU=1, Q=0., U=0., action=None),
                   O=dict(N_PLAYS=0, N_WINS=0, N_TIES=0, SCORE=0., PRIOR=1.,
                          C_PUCT=2., TAU=1., Q=0., U=0., action=None))


def main():
    """ Play two Monte Carlo Tree searches against one another...
    """
    # params:
    c = Config()
    g = Game()

    # Run one Game
    print("\nCurrent Player: {}".format(g.current_player.display))
    tree = MonteCarloTreeSearch(game=g, evaluation_func=random_value_policy,
                                node_param=NODE_PARAMS[g.current_player.display])
    tree.search(max_iterations=c.MCTS_ITERATIONS, max_runtime=c.MCTS_MAX_TIME)
    g.play(move=tree.recommended_play())
    tree.show_tree(level=1)
    g.show_board()

    while g.legal_plays():
        print("\nCurrent Player: {}".format(g.current_player.display))
        tree = MonteCarloTreeSearch(game=g, evaluation_func=random_value_policy,
                                    node_param=NODE_PARAMS[g.current_player.display])
        tree.search(max_iterations=c.MCTS_ITERATIONS, max_runtime=c.MCTS_MAX_TIME)
        g.play(move=tree.recommended_play())
        tree.show_tree(level=1)
        g.show_board()

    tree.show_tree(level=1)
    print("\nGame Winner")


if __name__ == '__main__':
    main()
