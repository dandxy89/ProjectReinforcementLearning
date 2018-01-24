# !/usr/bin/env python
# -*- coding: utf-8 -*-
""" TicTacToe.main
"""
import pickle

from RLBook.TicTacToe.Agent import RLAgent
from RLBook.TicTacToe.Environment import TicTacToeEnvironment
from RLBook.TicTacToe.RandomAgent import RLAgentRandom
from RLBook.TicTacToe.TicTacToe import TicTacToeGame


def trainer(trials):
    # Create two Action
    agent_one = RLAgentRandom(action_value=1)
    agent_two = RLAgent(action_value=-1, epsilon=0.75, step_size=0.1)

    # Create a Environment
    env = TicTacToeEnvironment(agent_one=agent_one, agent_two=agent_two, trials=trials)

    # Using a pre-trained Agent by reading in said Pickle File
    # with open('RLBook/TicTacToe/AGENT.pickle', 'rb') as handle:
    #     env.AGENT2 = pickle.load(handle)
    #     env.AGENT2.EPSILON = 0.4
    #     env.AGENT2.STEP_SIZE = 0.2

    # Train the Agents
    env.train()

    return dict(Agent1=env.AGENT1, Agent2=env.AGENT2), env.SCORES.copy()


if __name__ == '__main__':
    # Run the trainer
    n = 20000
    agents, scores = trainer(trials=n)
    print("Training Complete - Draws: {}, Wins (Agent1): {}, Wins (Agent2): {}\n".format(
        scores[0] / n, scores[1] / n, scores[-1] / n))

    # Continue Training and competing...
    agents["Agent2"].EPSILON, agents["Agent2"].STEP_SIZE = 0.4, 0.25
    n2 = 20000
    env2 = TicTacToeEnvironment(agent_one=agents["Agent1"], agent_two=agents["Agent2"], trials=n2)
    env2.train()

    # Continue Training and competing...
    agents["Agent2"].EPSILON, agents["Agent2"].STEP_SIZE = 0.45, 0.3
    n3 = 20000
    env3 = TicTacToeEnvironment(agent_one=agents["Agent1"], agent_two=env2.AGENT2, trials=n3)
    env3.train()

    # Writing a Pickle
    with open('RLBook/TicTacToe/AGENT.pickle', 'wb') as handle:
        pickle.dump(env3.AGENT2, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Read in said Pickle File
    with open('RLBook/TicTacToe/AGENT.pickle', 'rb') as handle:
        env3.AGENT2 = pickle.load(handle)

    # Human Interaction
    in_progress, active_gamer = True, env3.AGENT2
    active_gamer.EPSILON = 0  # Agent should make the decisions entirely on his own
    game = TicTacToeGame(columns=3, rows=3)

    while in_progress:  # Step through as opposed to using the Loop
        # Invoke the Agent
        game_state = game.get_state()
        eligible_actions = env2.get_eligible_indexes(array=game.BOARD)
        action = active_gamer.take_action(game_state, eligible_actions[0])
        game.update_board(action=action, value=active_gamer.ACTION)

        # Check if the End of the Game has been found?
        in_progress = not game.is_end_state()
        print(game.WINNER)

        # Show the Current State
        game.display()

        game.BOARD[0, 0] = 1
