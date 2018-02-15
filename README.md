
| Project                        | Created   | Updated    | Version |
|--------------------------------|-----------|------------|---------|
| Project Reinforcement Learning | 07/1/2017 | 15/02/2017 | 0.1.1   |

# Project Reinforcement Learning

### Overview

My solutions to various exercises from the the Reinforcement Learning book: An Introduction Book by Andrew Barto and Richard S. Sutton.

# Motivation

After reading Deep Thinking by Gary Kasparov - I know want to implement an Agent to play the game of Chess and to fully appreciate the the challenge of building an AI capable of doing so.

*   Revisiting Agent based systems and Reinforcement from University
*   Implement the algorithms from scratch
*   Eventually... get to the point where I can implement, appreciate and replicate to some degree the work by DeepMind, specifically AlplhaZero.

# End Goal

*   AlphaZero equivalent to play the game of Chess!

# Timeline

*   Jan -> Feb - implement various exercises from Chapters 1-5 and Chapter 8 AlphaZero TicTacToe
*   Mar -> Apr - implement various exercises from the remaining chapters 5-13
*   Apr -> May - commence the implementation of the Environment, Agent and auxiliary functionality to support the development work of the ChessZero Agent

# Chapter 8

One thing lead to another and I've implemented a AlphaZero styled implementation, applied to the game of TicTacToe.

Please not that to get the best performance its recommended to build tensorflow from the source.

# Bump Version

Easy bumping of a package version:

1.  ``` bumpversion --config-file .bumpversion.cfg major ``` - Example: 1.3.1 -> 2.0.0
2.  ``` bumpversion --config-file .bumpversion.cfg minor ``` - Example: 1.3.1 -> 1.4.0
3.  ``` bumpversion --config-file .bumpversion.cfg patch ``` - Example: 1.3.1 -> 1.3.2
