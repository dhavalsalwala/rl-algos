import gym
import matplotlib
import numpy as np

from agents.qlearning_agent import QLearningAgent

matplotlib.style.use('ggplot')


def play_episodes(env, n_episodes, Q):
    wins = 0
    total_reward = 0
    for episode in range(n_episodes):
        terminated = False
        state = env.reset()
        # print("*** Episode: ", episode)
        while not terminated:

            # Select best action to perform in a current state
            action = np.argmax(Q[state])

            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, info = env.step(action)

            # Summarize total reward
            total_reward += reward

            # Update current state
            state = next_state

            # Calculate number of wins over episodes
            if terminated and reward == 1.0:
                wins += 1
                break

    average_reward = total_reward / n_episodes

    return wins, total_reward, average_reward


# Load a Windy GridWorld environment
env_name = 'FrozenLake-v0'
env = gym.make(env_name)
agent = QLearningAgent(env_name, env, 50000, start_learning_rate=0.1, start_epsilon=1.0, discount_factor=0.95, decay_rate=0.001,
                       make_checkpoint=True, dir_location="/home/dsalwala/NUIG/Thesis/rl-algos/data")

Q = agent.load("/home/dsalwala/NUIG/Thesis/rl-algos/data/FrozenLake-v0_40000.npy")

# Search for a Q values
# Q, stats = agent.q_table, agent.stats

n_episodes = 1000
iteration_name = "QLearning"
wins, total_reward, average_reward = play_episodes(env, n_episodes, Q)

print(f'{iteration_name} :: number of wins over {n_episodes} episodes = {wins}')
print(f'{iteration_name} :: average reward over {n_episodes} episodes = {average_reward} \n\n')

# plotting.plot_episode_stats(stats)
