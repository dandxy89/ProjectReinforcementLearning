#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.Chapter8.DefaultPlayers

*   These players do not have Neural Networks.
*   Used as part of validating the functionality of the Game and MCTS prior to the inclusion of the NN

"""
from RLBook.Utils.Player import Player

# Default Players w/o Neural Networks
DEFAULT_PLAYERS = [Player(name='A', value=1, display='O'),
                   Player(name='B', value=-1, display='X')]
