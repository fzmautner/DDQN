## Double Deep Q-Network (DDQN)

Double Deep Q-Network (DDQN) is an enhancement to the original Deep Q-Network (DQN) algorithm, designed to address the overestimation bias problem in Q-learning. DDQN improves the stability and performance of reinforcement learning agents in various environments.

### Key Concepts

1. **Two Networks**: DDQN uses two neural networks - an online network and a target network. Both have the same architecture but different weights.

2. **Decoupled Action Selection and Evaluation**: The online network selects the best action, while the target network evaluates that action. This separation helps reduce overestimation bias.

3. **Periodic Updates**: The target network's weights are updated periodically with the online network's weights, allowing for more stable learning.

4. **Experience Replay**: A replay buffer is used to store and randomly sample past experiences, breaking correlation between consecutive samples, improving learning stability.

5. **Epsilon-Greedy Exploration**: The agent balances exploration and exploitation using a decaying epsilon-greedy policy, gradually reducing the exploration rate over time.

### Implementation Details

This is a simple implementation of DDQN for the Lunar Lander environment from OpenAI's RL Gym. The agent learns to control the lander, managing its thrusters to safely land on the moon's surface. The state space is continuous, while the action space is discrete, consisting of four possible actions.

### Requirements

- Python 3.x
- PyTorch
- Gym
- Numpy

### Usage

The file `DDQN.py` implements the model, which is run in the notebook `runDDQN.ipynb`. Functions are self-explanatory (i.e. `model.train()`, `model.run()`)
