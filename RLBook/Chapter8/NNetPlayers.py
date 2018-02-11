#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.Chapter8.NNetPlayers
"""
from RLBook.Utils.Player import Player

# Default Players w Neural Networks
NNetPlayers = [Player(name='A', value=1, display='O', use_nn=True),
               Player(name='B', value=-1, display='X', use_nn=True)]
