import gym
import numpy as np

from agents.dqn_agent import DQNAgent, RandomDQNAgent
from lib import plotting
from lib.utils.function_estimator import ANN


def play_episode(environment, nn, episodes):
    total_reward = 0
    total_epochs = 0

    for episode in range(episodes):

        state = environment.reset()
        terminated = False
        while not terminated:
            # environment.render()

            # Select best action to perform in a current state
            action = np.argmax(nn.predict(state))

            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, info = environment.step(action)

            # calculate total reward
            # print(reward)
            total_reward += reward
            total_epochs += 1

            # Update current state
            state = next_state

    print()
    print(f"Results after {episodes} episodes:")
    print(f"Average reward per episode: {total_reward / episodes}")
    print(f"Average time steps per episode: {total_epochs / episodes}")


# Load a Windy GridWorld environment
env_name = "CartPole-v0"
env = gym.make(env_name)

random_agent = RandomDQNAgent(env_name, env, 1000, is_state_box=True, memory_capacity=100000)
# random_agent.train()

agent = DQNAgent(env_name, env, 5000, learning_rate=0.00025, start_epsilon=1.0, discount_factor=0.99, decay_rate=0.0001,
                 make_checkpoint=True, is_state_box=True, batch_size=64, memory_capacity=100000)
# agent.memory = random_agent.memory
# agent.train()

weights, rewards, episode_len = agent.load("/home/dsalwala/NUIG/Thesis/rl-algos/data/CartPole-v0_1000 (1).npy")
stats = plotting.EpisodeStats(
    episode_lengths=episode_len,
    episode_rewards=rewards)

# Search for a Q values
# nn, stats = agent.nn, agent.stats

nn = ANN(4, 2, 0.00025)
nn.set_weights(weights)
play_episode(env, nn, 100)

env.close()

plotting.plot_episode_stats(stats)
