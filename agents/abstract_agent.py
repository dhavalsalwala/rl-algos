import os

import numpy as np

from lib import plotting


class BaseAgent:

    def __init__(self, env_name, env, num_episodes, n_step=1, discount_factor=0.95, start_learning_rate=0.1, start_epsilon=1.0,
                 decay_rate=0.001, action_space_n=None, render_env=False, make_checkpoint=False, is_state_box=False):

        self.env_name = env_name
        self.env = env
        self.MAX_STEPS = 200
        self.num_episodes = num_episodes
        self.start_learning_rate = start_learning_rate
        self.learning_rate = 0
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
        self.stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))

        if action_space_n is None:
            self.nA = env.action_space.n
        else:
            self.nA = action_space_n

        if self.is_state_box:
            self.nS = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
            self.nS = np.round(self.nS, 0).astype(int) + 1

    def _learn(self):
        raise NotImplementedError

    def train(self, title=None, version=None):
        print("\nEnvironment\n--------------------------------")
        print("Name :", self.env_name)
        print("\nObservation\n--------------------------------")
        if self.is_state_box:
            print("Shape :", self.nS[0], " X ", self.nS[1])
        else:
            print("Shape :", self.env.observation_space.n)
        print("\nAction\n--------------------------------")
        print("Shape :", self.env.action_space.n, "\n")
        if title is not None and version is not None:
            print("\nTraining started using " + title + " Agent v"+str(version)+".\n")
        else:
            print("\nTraining started.\n")

    def _get_action(self):
        raise NotImplementedError

    def score(self, n_episode):
        num = round(self.num_episodes / 50)
        if n_episode % num == 0 and n_episode != 0:
            avg = np.mean(self.stats.episode_rewards[(n_episode + 1 - num):(n_episode + 1)])
            print("  |  Avg Reward (last " + str(num) + ")=", avg, " | Total Avg Reward =",
                  np.sum(self.stats.episode_rewards) / n_episode, " | Epsilon =", self.epsilon)

    def _update_statistics(self, R, time_step, i_episode):
        self.stats.episode_rewards[i_episode] += R
        self.stats.episode_lengths[i_episode] = time_step

    def save(self, data, i_episode, force_save=False):
        num = round(self.num_episodes / 5)
        if force_save or (i_episode % num == 0 and i_episode != 0):
            file_path = os.path.join(self.dir_location, self.env_name + '_' + str(i_episode) + '.npy')
            np.save(file_path, data)
            print("\nSaved checkpoint to: ", file_path, "\n")

    @staticmethod
    def load(file):
        return np.load(file)

    @staticmethod
    def exit(data, message):
        print("\n\n"+message+"\n")
        print("\nQ Table\n--------------------------------")
        print(data)
