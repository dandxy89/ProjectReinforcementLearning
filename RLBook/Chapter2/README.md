
| Project                        | Created   | Updated    | Version |
|--------------------------------|-----------|------------|---------|
| Project Reinforcement Learning | 07/1/2017 | 24/1/2017  | 0.1.0   |

# Overview

This folder contains all the functionality to train various Agents for the purpose of different Policies.

# Directory

*   [Bandits.py](RLBook/Chapter2/Bandits.py) - Single, Multi and Binary Armed Bandits
*   [EGreedy.py](RLBook/Chapter2/EGreedy.py) - epsilon greedy policy
*   [Incremental.py](RLBook/Chapter2/Incremental.py) - Incremental update policy
*   [LinearRewardInaction.py](RLBook/Chapter2/LinearRewardInaction.py) - Linear Reward Inaction (Binary armed agent)
*   [LinearRewardPenalty.py](RLBook/Chapter2/LinearRewardPenalty.py) - Linear Reward Penalty (Binary armed agent)
*   [main.py](RLBook/Chapter2/main.py) - Training script
*   [ModellingEnvironment.py](RLBook/Chapter2/ModellingEnvironment.py) - Environment handling class
*   [Nonstationary.py](RLBook/Chapter2/Nonstationary.py) - Non-stationary policy Agent
*   [Pursuit.py](RLBook/Chapter2/Pursuit.py) - Pursuit policy Agent
*   [Softmax.py](RLBook/Chapter2/Softmax.py) - Softmax policy Agent

# Results

Varying the Agent type has quite an impact on the performance of the Agent.

For this I consider a correct decision where the return of a Bandit to be positive, other implementations use different variations of this.

As a opening chapter this is both interesting to implement and do optimally. Highly recommended...
 