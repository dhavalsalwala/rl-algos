import math
import os
import time

import numpy as np

from modules.lib import plotting


class BaseAgent:

    def __init__(self, env_name, env, num_episodes, n_step=1, discount_factor=0.95, learning_rate=0.01, start_learning_rate=0.1, start_epsilon=1.0,
                 decay_rate=0.001, action_space_n=None, render_env=False, make_checkpoint=False, is_state_box=False, batch_size=25,
                 memory_capacity=1000):

        self.start_time = 0
        self.env_name = env_name
        self.env = env
        self.MAX_STEPS = 200
        self.num_episodes = num_episodes
        self.start_learning_rate = start_learning_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.start_epsilon = start_epsilon
        self.epsilon = 0
        self.decay_rate = decay_rate
        self.make_checkpoint = make_checkpoint
        self.n_step = n_step
        self.dir_location = "/home/dsalwala/NUIG/Thesis/rl-algos/data"
        self.is_state_box = is_state_box
        self.action_space_n = action_space_n
        self.render_env = render_env
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))

        if action_space_n is None:
            self.nA = env.action_space.n
        else:
            self.nA = action_space_n

        if self.is_state_box:
            self.nS = self.env.observation_space.shape[0]
        else:
            self.nS = 1

    def _learn(self):
        raise NotImplementedError

    def train(self, title=None, version=None):
        print("\nEnvironment\n--------------------------------")
        print("Name :", self.env_name)
        print("\nObservation\n--------------------------------")
        if self.is_state_box:
            print("Shape : [", self.nS, ",]")
        else:
            print("Shape :", self.nS)
        print("\nAction\n--------------------------------")
        print("Shape :", self.nA, "\n")
        if title is not None and version is not None:
            print("\nTraining started using " + title + " Agent v"+str(version)+".\n")
        else:
            print("\nTraining started.\n")

        self.start_time = time.time()

    def _get_action(self):
        raise NotImplementedError

    def score(self, n_episode):
        num = math.ceil(self.num_episodes * 0.02)
        if n_episode != 0 and n_episode % num == 0:
            avg = np.mean(self.stats.episode_rewards[(n_episode + 1 - num):(n_episode + 1)])
            print("  |  Avg Reward (last " + str(num) + ")=", avg, " | Total Avg Reward =",
                  np.sum(self.stats.episode_rewards[1:]) / n_episode, " | Epsilon =", self.epsilon)

    def _update_statistics(self, R, time_step, i_episode):
        self.stats.episode_rewards[i_episode] += R
        self.stats.episode_lengths[i_episode] = time_step

    def save(self, i_episode, data=None, force_save=False, is_tensor=False, is_a2c=False):
        num = math.ceil(self.num_episodes * 0.2)
        if (force_save or (i_episode % num == 0 and i_episode != 0)) and not is_tensor:
            file_path = os.path.join(self.dir_location, self.env_name + '_' + str(i_episode) + '.npy')
            np.save(file_path, (data if data is not None else self.nn.get_weights(), self.stats.episode_rewards, self.stats.episode_lengths))
            print("\nSaved checkpoint to: ", file_path, "\n")
        elif (force_save or (i_episode % num == 0 and i_episode != 0)) and is_tensor:
            file_path = os.path.join(self.dir_location + '/model', self.env_name + '_model_' + str(i_episode) + '.ckpt')
            file_path_stats = os.path.join(self.dir_location, self.env_name + '_stats_' + str(i_episode) + '.npy')
            if is_a2c:
                self.actor_nn.save(file_path)
            else:
                self.policy_nn.save(file_path)
            np.save(file_path_stats, (self.stats.episode_rewards, self.stats.episode_lengths))
            print("\nSaved checkpoint to: ", file_path, "\n")

    @staticmethod
    def load(file):
        return np.load(file)

    def exit(self, message):
        print("Training Completed.\n--------------------------------")
        print("Total Time: " + str(time.time() - self.start_time) + " seconds.")
        print(message + "\n")
