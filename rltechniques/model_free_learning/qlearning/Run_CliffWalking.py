import sys
import time
from lib import plotting
import numpy as np
from lib.envs.cliff_walking import CliffWalkingEnv
from agents.qlearning_agent import QLearningAgent


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
agent = QLearningAgent(environment, 200)
agent.train()


# Search for a Q values
Q, stats = agent.q_table, agent.stats

play_episode(environment, Q)

plotting.plot_episode_stats(stats)
