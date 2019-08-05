import time
from collections import namedtuple

import numpy as np
import rllab.plotter as plotter
from rllab.algos.base import RLAlgorithm
from rllab.misc import logger


class BatchMADQNKERAS(RLAlgorithm):
    """
    Base class for batch sampling-based deep q network.
    """

    def __init__(self, env, networks, scope=None, n_itr=5000,
                 start_itr=0, batch_size=32, max_path_length=200,
                 plot=False, max_epsilon=1, min_epsilon=0.01,
                 pre_trained_size=10000,
                 save_param_update=125, **kwargs):
        """
        :param env: Environment
        :param policy: Policy
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :return:
        """
        self.env = env
        self.nS = env.observation_space[0]
        self.nA = env.action_space[0].n
        self.networks = networks
        self.random_seed = 2
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.max_epsilon = max_epsilon
        self.render = False
        self.min_epsilon = min_epsilon
        self.save_param_update = save_param_update
        self.pre_trained_size = pre_trained_size
        EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards", "episode_losses", "episode_epsilon"])
        self.stats = EpisodeStats(
            episode_lengths=[],
            episode_rewards=[],
            episode_losses=[],
            episode_epsilon=[])

        self.init_opt()

    def run_random_exp(self):

        observations = self.env.reset()
        for itr in range(self.pre_trained_size):

            actions = []
            one_hot_actions = []
            for agent_index in range(self.env.n):
                action = self.networks[agent_index].get_action(observations[agent_index], force_random=True)
                speed = 0.9 if self.env.agents[agent_index].adversary else 1
                one_hot_action = np.zeros(self.nA)
                one_hot_action[action] = speed
                one_hot_actions.append(one_hot_action)
                actions.append(action)

            next_observations, R, done, info = self.env.step(one_hot_actions)

            for agent_index in range(self.env.n):

                if done[agent_index]:
                    R[agent_index] -= 50

                # Remember experience
                self.networks[agent_index].add_to_memory(observations[agent_index], actions[agent_index], R[agent_index],
                                                         next_observations[agent_index], done[agent_index])

            if any(done):
                observations = self.env.reset()
            else:
                observations = next_observations

    def train(self):

        logger.log("Populating replay memory with random experience...")
        self.run_random_exp()

        start_time = time.time()
        total_time_step = 0
        for itr in range(self.start_itr, self.n_itr + 1):
            itr_start_time = time.time()

            self.stats.episode_lengths.append(itr)
            self.stats.episode_lengths[itr] = []
            self.stats.episode_rewards.append(itr)
            self.stats.episode_rewards[itr] = []
            self.stats.episode_losses.append(itr)
            self.stats.episode_losses[itr] = []
            self.stats.episode_epsilon.append(itr)

            with logger.prefix('itr #%d | ' % itr):

                observations = self.env.reset()
                logger.log("Running trajectories, obtaining samples and optimizing Q network...")
                for time_step in range(self.max_path_length):

                    total_time_step += 1

                    # render environment
                    if self.render:
                        self.env.render()
                        time.sleep(0.5)

                    # Take a next step and next action
                    actions = []
                    one_hot_actions = []
                    for agent_index in range(self.env.n):
                        action = self.networks[agent_index].get_action(observations[agent_index])
                        speed = 0.9 if self.env.agents[agent_index].adversary else 1
                        one_hot_action = np.zeros(self.nA)
                        one_hot_action[action] = speed
                        one_hot_actions.append(one_hot_action)
                        actions.append(action)

                    next_observations, R, done, info = self.env.step(one_hot_actions)

                    step_losses = np.zeros(self.env.n)
                    for agent_index in range(self.env.n):

                        if done[agent_index]:
                            R[agent_index] -= 50

                        # Remember experience
                        self.networks[agent_index].add_to_memory(observations[agent_index], actions[agent_index], R[agent_index],
                                                                 next_observations[agent_index], done[agent_index])

                        # Replay and train memories
                        history = self.networks[agent_index].replay(total_time_step)
                        step_losses[agent_index] += history.history["loss"][0]

                    observations = next_observations
                    self.stats.episode_rewards[itr].append(R)
                    self.stats.episode_lengths[itr].append(time_step)
                    self.stats.episode_losses[itr].append(step_losses)

                    if any(done):
                        break

                self.stats.episode_epsilon[itr] = [self.networks[i].epsilon for i in range(self.env.n)]

                logger.log("Logging statistics...")
                self.log_statistics(itr, time_step + 1)

                if True:
                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr)
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")

                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)

    def log_statistics(self, itr, total_steps):

        rewards = np.sum(np.array(self.stats.episode_rewards[itr]), axis=0) / total_steps
        loss = np.sum(np.array(self.stats.episode_losses[itr]), axis=0) / total_steps
        logger.record_tabular('Iteration', itr)
        logger.record_tabular('NofTrajectories', total_steps)
        logger.record_tabular('Epsilon', np.array(self.stats.episode_epsilon[itr]))
        logger.record_tabular('Return', rewards)
        logger.record_tabular('MaxReturn', np.max(rewards))
        logger.record_tabular('MinReturn', np.min(rewards))
        logger.record_tabular("Loss", loss)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.networks, self.max_path_length)
