### Overview

This repository provides solutions to the most popular Reinforcement Learning algorithms. These are meant to serve as a learning tool to complement the theoretical materials from

- [Reinforcement Learning: An Introduction (2nd Edition)](http://incompleteideas.net/book/RLbook2018.pdf)
- [David Silver's Reinforcement Learning Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

All code is written in Python 3 and uses RL environments from [OpenAI Gym](https://gym.openai.com/). Advanced techniques use [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for neural network implementations.

### Reinforcement Techniques

  #### Multi Agent RL
  
  - Environment info
    - Pursuit Evasion
  
  - Multi Agent Deep Q Network
    - [Pursuit Evasion](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/multi_agent/dqn/run_pursuit.py)

  #### Single Agent RL

- Dynamic Programming
  - [Policy and Value Iteration- Frozen Lake](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/single_agent/dynamic_programming/Run_FrozenLake.py)
- Model Free Learning
  - Sarsa
    - [Cliff Walking](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/single_agent/model_free_learning/sarsa/Run_CliffWalking.py)
    - [Frozen Lake](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/single_agent/model_free_learning/sarsa/Run_FrozenLake.py)
    - [Grid World](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/single_agent/model_free_learning/sarsa/Run_GridWorld.py)
    - [Windy Grid World](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/single_agent/model_free_learning/sarsa/Run_WindyGridWorld.py)
  - Q Learning
    - [Cliff Walking](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/single_agent/model_free_learning/qlearning/Run_CliffWalking.py)
    - [Frozen Lake](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/single_agent/model_free_learning/qlearning/Run_FrozenLake.py)
    - [Mountain Car](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/single_agent/model_free_learning/qlearning/Run_MountainCar.py)
    - [Taxi Ride](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/single_agent/model_free_learning/qlearning/Run_TaxiRide.py)
  - Deep Q Network (Fixed Q values and Double DQN)
    - [Cart Pole](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/single_agent/model_free_learning/dqn/Run_CartPole.py)
    - [Lunar Lander](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/single_agent/model_free_learning/dqn/Run_LunarLander.py)
- Policy Gradient
  - Monte Carlo Policy Gradient (REINFORCE)
    - [Cart Pole](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/single_agent/policy_gradient/Run_CartPole.py)
  - Actor Critic Policy Gradient (A2C)
    - [Cart Pole](https://github.com/dhavalsalwala/rl-algos/blob/master/rltechniques/single_agent/policy_gradient/Run_CartPole_A2C.py)

### References

Textbooks:

- [Reinforcement Learning: An Introduction (2nd Edition)](http://incompleteideas.net/book/RLbook2018.pdf)

Youtube Lecture Series:

- [Introduction to Reinforcement Learning by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

MOOC:

- [School of AI's Free Course - Move37](https://www.theschool.ai/courses/move-37-course/)

Classes:

- [David Silver's Reinforcement Learning Course (UCL, 2015)](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- [CS294 - Deep Reinforcement Learning (Berkeley, Fall 2015)](http://rll.berkeley.edu/deeprlcourse/)

GitHub Resources:

- I highly recommend School of AI's Free Course [Move37](https://www.theschool.ai/courses/move-37-course/)
GitHub: https://github.com/colinskow/move37
- [Multi-agent deep reinforcement learning library (MADRL).](https://github.com/sisl/MADRL) - by Stanford Intelligent Systems Laboratory
- [Custom Environments based on OpenAI gym](https://github.com/dennybritz/reinforcement-learning/tree/master/lib) - by Denny Britz

### Contact

- If you have a suggestion or you came across any bug/issue, shoot a mail at mailto:dhavalsalwala@gmail.com
