import gym
import numpy as np

from agents.qlearning_2dbox_agent import QLearningAgent


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
env_name = "MountainCar-v0"
env = gym.make(env_name)
agent = QLearningAgent(env_name, env, 50000, start_learning_rate=0.1, start_epsilon=1.0, discount_factor=0.95, decay_rate=0.001,
                       make_checkpoint=True, is_state_box=True)
# agent.train()
Q = agent.load("/home/dsalwala/NUIG/Thesis/rl-algos/data/MountainCar-v0_30000.npy")

# Search for a Q values
# Q, stats = agent.q_table, agent.stats

play_episode(env, Q)

env.close()

# plotting.plot_episode_stats(stats)
