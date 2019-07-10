### Overview

This repository provides code, exercises and solutions for popular Reinforcement Learning algorithms. These are meant to serve as a learning tool to complement the theoretical materials from

- [Reinforcement Learning: An Introduction (2nd Edition)](http://incompleteideas.net/book/RLbook2018.pdf)
- [David Silver's Reinforcement Learning Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

All code is written in Python 3 and uses RL environments from [OpenAI Gym](https://gym.openai.com/). Advanced techniques use [Tensorflow](https://www.tensorflow.org/) for neural network implementations.

### Reinforcement Techniques:

- Dynamic Programming
  - [Policy and Value Iteration- Frozen Lake](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/dynamic_programming/Run_FrozenLake.py)
- Model Free Learning
  - Sarsa
    - [Cliff Walking](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/model_free_learning/sarsa/Run_CliffWalking.py)
    - [Frozen Lake](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/model_free_learning/sarsa/Run_FrozenLake.py)
    - [Grid World](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/model_free_learning/sarsa/Run_GridWorld.py)
    - [Windy Grid World](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/model_free_learning/sarsa/Run_WindyGridWorld.py)
  - Q Learning
    - [Cliff Walking](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/model_free_learning/qlearning/Run_CliffWalking.py)
    - [Frozen Lake](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/model_free_learning/qlearning/Run_FrozenLake.py)
    - [Mountain Car](hhttps://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/model_free_learning/qlearning/Run_MountainCar.py)
    - [Taxi Ride](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/model_free_learning/qlearning/Run_TaxiRide.py)
  - Deep Q Network
    - [Cart Pole](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/model_free_learning/dqn/Run_CartPole.py)

### References

Textbooks:

- [Reinforcement Learning: An Introduction (2nd Edition)](http://incompleteideas.net/book/RLbook2018.pdf)

Classes:

- [David Silver's Reinforcement Learning Course (UCL, 2015)](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- [CS294 - Deep Reinforcement Learning (Berkeley, Fall 2015)](http://rll.berkeley.edu/deeprlcourse/)

Lib (Custom Environments based on OpenAI gym)

- https://github.com/dennybritz/reinforcement-learning
