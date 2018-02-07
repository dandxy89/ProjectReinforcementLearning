#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" RLBook.Chapter7

-   Python Implementation of the RL Book Exercises
-   Planning

"""

__version__ = "0.1.0"
__developers__ = "Dan Dixey (2018)"

DEFAULT_NODE_PARAMS = dict(N_PLAYS=0, N_WINS=0, N_TIES=0, SCORE=0., PRIOR=1., C_PUCT=2, TAU=1., Q=0., U=0.,
                           action=None)
