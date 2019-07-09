import gym
import numpy as np

from agents.qlearning_agent import QLearningAgent


def play_episode(environment, Q, episodes):

    total_reward = 0
    penalties = 0
    total_epochs = 0

    for episode in range(episodes):

        state = environment.reset()
        terminated = False

        while not terminated:
            # time.sleep(1)
            # env.render()

            # Select best action to perform in a current state
            action = np.argmax(Q[state])

            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, info = environment.step(action)

            # calculate total reward
            # print(reward)
            total_reward += reward
            total_epochs += 1
            if reward == -10:
                penalties += 1

            # Update current state
            state = next_state

    print()
    print(f"Results after {episodes} episodes:")
    print(f"Average reward per episode: {total_reward / episodes}")
    print(f"Average time steps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {penalties / episodes}")


# Load a Windy GridWorld environment
env_name = 'Taxi-v2'
env = gym.make(env_name)
agent = QLearningAgent(env_name, env, 10000, start_learning_rate=0.1, start_epsilon=1.0,
                       discount_factor=0.95, decay_rate=0.001, make_checkpoint=True)
agent.train()
# Q = agent.load("/home/dsalwala/NUIG/Thesis/rl-algos/data/Taxi-v2_10000.npy")

# Search for a Q values
Q, stats = agent.q_table, agent.stats

play_episode(env, Q, 100)

# plotting.plot_episode_stats(stats)
