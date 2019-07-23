import sys

import gym
import numpy as np
import tensorflow as tf

from agents.reinforce_agent import REINFORCEAgent
from modules.lib import plotting


def play_episode(environment, nn, episodes):
    total_reward = 0
    total_epochs = 0

    for episode in range(episodes):

        state = environment.reset()
        terminated = False
        while not terminated:
            environment.render()
            print("\rPlaying Episode {}/{}".format(episode + 1, episodes), end="")
            sys.stdout.flush()

            # Select best action to perform in a current state
            state = state.reshape(1, len(state))
            action_prob = nn.predict(state).flatten()
            action = np.random.choice(len(action_prob), 1, p=action_prob)[0]

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
with tf.Session() as session:
    train = True
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    agent = REINFORCEAgent(env_name, env, 300, session, learning_rate=0.01, discount_factor=0.95, make_checkpoint=True,
                           is_state_box=True, batch_size=64)

    if train:
        agent.train()
        stats = agent.stats
        nn = agent.policy_nn
    else:
        nn = agent.policy_nn
        nn.load("/home/dsalwala/NUIG/Thesis/rl-algos/data/model/CartPole-v0_model_3000.ckpt")
        rewards, episode_len = agent.load("/home/dsalwala/NUIG/Thesis/rl-algos/data/CartPole-v0_stats_3000.npy")
        stats = plotting.EpisodeStats(
            episode_lengths=episode_len,
            episode_rewards=rewards)

    play_episode(env, nn, 1)
    env.close()
    plotting.plot_episode_stats(stats)
