# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" TicTacToe.main
"""
from RLBook.TicTacToe.Agent import RLAgent
from RLBook.TicTacToe.Environment import TicTacToeEnvironment


def trainer():
    # Create two Action
    agent_one = RLAgent(action_value=1)
    agent_two = RLAgent(action_value=-1)

    # Create a Environment
    env = TicTacToeEnvironment(agent_one=agent_one, agent_two=agent_two, trials=10000)

    # Train the Agents
    env.train()

    return dict(Agent1=env.AGENT1, Agent2=env.AGENT2), env.SCORES, env


if __name__ == '__main__':
    # Run the trainer
    agents, scores, env = trainer()

    # print(agents["Agent1"].POLICY)
    print(scores)

    # self = env
