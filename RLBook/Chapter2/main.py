# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.Chapter2.main

*   Script for Running different experiments...

Info
----

Testable so far:

*   e-Greedy
*   Softmax
*   Linear, reward-penalty
*   Linear, reward-inaction
*   Incremental

TODO:

*   Non-stationary
*   Optimistic Initial Values
*   Pursuit

"""
from RLBook.Chapter2.ModellingEnvironment import ModelEnvironment
from RLBook.Utils.PolicyTypes import PolicyEnum

if __name__ == '__main__':
    # params:
    trials = 20000

    # Initialise a e-Greedy Environment
    # env1 = ModelEnvironment(trials=trials, bandits=10, policy=PolicyEnum.EGREEDY,
    #                         epsilons=[0, 0.1, 0.5, 0.75])
    # env1.run()
    # env1.print_results()
    # env1.generate_charts()
    # MAX: 15962

    # Initialise a Softmax Environment
    # env2 = ModelEnvironment(trials=trials, bandits=10, policy=PolicyEnum.SOFTMAX,
    #                         temperatures=[0.000001, 0.1, 0.5, 0.75])
    # env2.run()
    # env2.print_results()
    # env2.generate_charts()
    # MAX: 16248

    # Initialise a Linear, reward-penalty Policy
    # env3 = ModelEnvironment(trials=trials, bandits=1, policy=PolicyEnum.LINEAR_REWARD_PENALTY,
    #                         epsilons=[0, 0.1, 0.5, 0.75], alpha=0.1)
    # env3.run()
    # env3.print_results()
    # env3.generate_charts()

    # Initialise a Linear, reward-inaction Policy
    # env4 = ModelEnvironment(trials=trials, bandits=1, policy=PolicyEnum.LINEAR_REWARD_INACTION,
    #                         epsilons=[0, 0.1, 0.5, 0.75], alpha=0.1, probability=0.2)
    # env4.run()
    # env4.print_results()
    # env4.generate_charts()

    # Initialise a Incremental Policy
    # env5 = ModelEnvironment(trials=trials, bandits=10, policy=PolicyEnum.INCREMENTAL,
    #                         epsilons=[0, 0.001, 0.01, 0.1, 0.2])
    # env5.run()
    # env5.print_results()
    # env5.generate_charts()
    # MAX: 16191

    # Initialise a Nonstationary Policy
    # env6 = ModelEnvironment(trials=trials, bandits=10, policy=PolicyEnum.NON_STATIONARY,
    #                         beta=0.1, epsilons=[0, 0.001, 0.01, 0.1, 0.2])
    # env6.run()
    # env6.print_results()
    # env6.generate_charts()
    # # MAX: TODO

    # Initialise a Pursuit Policy
    # env7 = ModelEnvironment(trials=trials, bandits=10, policy=PolicyEnum.PURSUIT)
    # env7.run()
    # env7.print_results()
    # env7.generate_charts()
    # MAX: 12546

    # Initialise the UCB variation
    env6 = ModelEnvironment(trials=trials, bandits=10, policy=PolicyEnum.UCB,
                            beta=0.1, epsilons=[0, 0.01, 0.09, 0.095, 0.6])
    env6.run()
    env6.print_results()
    env6.generate_charts()
    # MAX: 16019
