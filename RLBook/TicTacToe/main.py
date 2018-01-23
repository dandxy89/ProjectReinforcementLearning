# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" TicTacToe.main
"""

from RLBook.TicTacToe.Agent import RLAgent
from RLBook.TicTacToe.AgentTwo import RLAgent2
from RLBook.TicTacToe.Environment import TicTacToeEnvironment
from RLBook.TicTacToe.TicTacToe import TicTacToeGame


def trainer(trials):
    # Create two Action
    agent_one = RLAgent(action_value=1, epsilon=0.99, step_size=0.1)
    agent_two = RLAgent2(action_value=-1, epsilon=0.6, step_size=0.5)

    # Create a Environment
    env = TicTacToeEnvironment(agent_one=agent_one, agent_two=agent_two, trials=trials)

    # Train the Agents
    env.train()

    return dict(Agent1=env.AGENT1, Agent2=env.AGENT2), env.SCORES, env


if __name__ == '__main__':
    # Run the trainer
    n = 200
    agents, scores, env = trainer(trials=n)
    print("Training Complete - Draws: {}, Wins (Agent1): {}, Wins (Agent2): {}\n".format(scores[0] / n,
                                                                                         scores[1] / n,
                                                                                         scores[-1] / n))

    #  Continue Training and competing...
    agents["Agent1"].UPDATE = False
    agents["Agent2"].EPSILON = 0.4
    agents["Agent2"].STEP_SIZE = 0.3
    n2 = 200000
    env = TicTacToeEnvironment(agent_one=agents["Agent1"], agent_two=agents["Agent2"], trials=n2)
    env.train()

    print("Training Phase 2 Complete - Draws: {}, Wins (Agent1): {}, Wins (Agent2): {}".format(env.SCORES[0] / n2,
                                                                                               env.SCORES[1] / n2,
                                                                                               env.SCORES[-1] / n2))

    # TODO: Add a Human into the Testing Loop (bit too fiddly)
    self = env
    in_progress = True
    active_gamer = self.AGENT2
    game = TicTacToeGame(columns=self.DEFAULT_BOARD[1], rows=self.DEFAULT_BOARD[0])

    # Invoke the Agent
    game_state = game.get_state()
    eligible_actions = self.get_eligible_indexes(array=game.BOARD)
    action = active_gamer.take_action(game_state, eligible_actions[0])
    game.update_board(action=action, value=active_gamer.ACTION)

    game.display()

    game.is_end_state()
    game.WINNER

    game.BOARD[2, 0] = 1
