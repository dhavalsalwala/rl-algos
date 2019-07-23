import gym
import numpy as np
from modules.lib import plotting
from agents.sarsa_agent import SARSAAgent
import matplotlib

matplotlib.style.use('ggplot')


def play_episodes(environment, n_episodes, Q):
    wins = 0
    total_reward = 0
    for episode in range(n_episodes):
        terminated = False
        state = environment.reset()
        print("*** Episode: ", episode)
        while not terminated:

            # Select best action to perform in a current state
            action = np.argmax(Q[state])

            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, info = environment.step(action)

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
environment = gym.make('FrozenLake-v0')
agent = SARSAAgent(environment, 1000, n_step=2, learning_rate=0.01, epsilon=0.8)
agent.train()

# Search for a Q values
Q, stats = agent.q_table, agent.stats  # NStepSarsa.apply(4, environment, 1000, alpha=0.01, epsilon=0.8)

print(Q)

n_episodes = 1000
iteration_name = "SARSA"
wins, total_reward, average_reward = play_episodes(environment, n_episodes, Q)

print(f'{iteration_name} :: number of wins over {n_episodes} episodes = {wins}')
print(f'{iteration_name} :: average reward over {n_episodes} episodes = {average_reward} \n\n')

plotting.plot_episode_stats(stats)
