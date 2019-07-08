from lib import plotting
import numpy as np


class BaseAgent:

    def __init__(self, env, num_episodes, n_step=1, discount_factor=1.0, learning_rate=0.5, epsilon=0.1,
                 min_epsilon=0, action_space_n=None, render_env=False):

        self.env = env
        self.num_episodes = num_episodes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.n_step = n_step
        self.action_space_n = action_space_n
        self.render_env = render_env
        self.stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))

        if action_space_n is None:
            self.nA = env.action_space.n
        else:
            self.nA = action_space_n

    def _learn(self):
        raise NotImplementedError

    def train(self, title=None, version=None):
        if title is not None and version is not None:
            print("\nTraining started using " + title + " Agent v"+str(version)+".\n")
        else:
            print("\nTraining started.\n")

    def _get_action(self):
        raise NotImplementedError

    def accuracy(self):
        raise NotImplementedError

    def _update_statistics(self, R, time_step, i_episode):
        self.stats.episode_rewards[i_episode] += R
        self.stats.episode_lengths[i_episode] = time_step

    @staticmethod
    def exit(message):
        print("\n\n"+message+"\n")
