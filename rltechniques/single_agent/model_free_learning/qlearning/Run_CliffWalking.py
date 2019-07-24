import sys
import time

import numpy as np

from agents.qlearning_agent import QLearningAgent
from modules.lib import plotting
from modules.lib.envs.cliff_walking import CliffWalkingEnv


def play_episode(env, Q):
    terminated = False
    state = env.reset()
    while not terminated:

        time.sleep(1)
        sys.stdout.flush()
        env.render()

        # Select best action to perform in a current state
        action = np.argmax(Q[state])

        # Perform an action an observe how environment acted in response
        next_state, reward, terminated, info = env.step(action)

        # Update current state
        state = next_state

        # Calculate number of wins over episodes
        if terminated and reward == 1.0:
            break


# Load a Windy GridWorld environment
environment = CliffWalkingEnv()
agent = QLearningAgent("CliffWalking-v0", environment, 1000, start_learning_rate=0.1, start_epsilon=1.0,
                       discount_factor=0.95, decay_rate=0.001, make_checkpoint=True)
# agent.train()
Q, rewards, episode_len = agent.load("/home/dsalwala/NUIG/Thesis/rl-algos/data/CliffWalking-v0_1000.npy")
stats = plotting.EpisodeStats(
    episode_lengths=episode_len,
    episode_rewards=rewards)

# Search for a Q values
# Q, stats = agent.q_table, agent.stats

play_episode(environment, Q)

plotting.plot_episode_stats(stats)
