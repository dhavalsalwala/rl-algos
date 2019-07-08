import sys
import time
from lib import plotting
import numpy as np
from lib.envs.grid_world import GridworldEnv
from agents.sarsa_agent import SARSAAgent


def play_episode(environment, Q):
    terminated = False
    state = environment.reset()
    while not terminated:

        time.sleep(1)
        sys.stdout.flush()
        environment.render()

        # Select best action to perform in a current state
        action = np.argmax(Q[state])

        # Perform an action an observe how environment acted in response
        next_state, reward, terminated, info = environment.step(action)

        # Update current state
        state = next_state

        # Calculate number of wins over episodes
        if terminated and reward == 1.0:
            break


# Load a Windy GridWorld environment
env = GridworldEnv()
agent = SARSAAgent(env, 200)
agent.train()

# Search for a Q values
Q, stats = agent.q_table, agent.stats

play_episode(env, Q)

plotting.plot_episode_stats(stats)
