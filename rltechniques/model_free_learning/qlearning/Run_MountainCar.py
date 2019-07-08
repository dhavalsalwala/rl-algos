import gym
import numpy as np

from agents.qlearning_2dbox_agent import QLearningAgent
from lib import plotting


def play_episode(environment, Q):
    terminated = False
    state = environment.reset()

    while not terminated:

        state = (state - environment.observation_space.low) * np.array([10, 100])
        state = np.round(state, 0).astype(int)

        environment.render()

        # Select best action to perform in a current state
        action = np.argmax(Q[state[0]][state[1]])

        # Perform an action an observe how environment acted in response
        next_state, reward, terminated, info = environment.step(action)

        # Update current state
        state = next_state

        # Calculate number of wins over episodes
        if terminated and reward == 1.0:
            break


# Load a Windy GridWorld environment
environment = gym.make("MountainCar-v0")
agent = QLearningAgent(environment, 5000, discount_factor=0.9, learning_rate=0.2, epsilon=0.8)
agent.train()

# Search for a Q values
Q, stats = agent.q_table, agent.stats

play_episode(environment, Q)

environment.close()

plotting.plot_episode_stats(stats)
