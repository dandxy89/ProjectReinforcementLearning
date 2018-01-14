# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.nArmedBandit.main

*   Script for Running different experiments...

Info
----

Testable so far:

*   e-Greedy
*   Softmax

"""
from RLBook.nArmedBandit.Extras import PolicyEnum
from RLBook.nArmedBandit.ModellingEnvironment import ModelEnvironment

if __name__ == '__main__':
    # params:
    bandits = 10
    trials = 20000

    # Initialise a e-Greedy Environment
    env1 = ModelEnvironment(trials=trials, bandits=bandits, policy=PolicyEnum.EGREEDY,
                            epsilons=[0, 0.1, 0.5, 0.75])
    env1.run()
    env1.print_results()
    env1.generate_charts()
    # Positive Rewards Achieved:  [11566 15962 14627 13938]

    # Initialise a Softmax Environment
    # env2 = ModelEnvironment(trials=trials, bandits=bandits, policy=PolicyEnum.SOFTMAX,
    #                         temperatures=[0.000001, 0.1, 0.5, 0.75])
    # env2.run()
    # env2.print_results()
    # env2.generate_charts()
    # [10095 15768 16248 15537]
