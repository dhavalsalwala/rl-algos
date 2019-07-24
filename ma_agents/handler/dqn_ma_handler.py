import time

import numpy as np
import rllab.plotter as plotter
import tensorflow as tf
from rllab.algos.base import RLAlgorithm
from rllab.misc import logger
from rllab.sampler.stateful_pool import ProgBarCounter
from sandbox.rocky.tf.policies.base import Policy
from ma_agents.sampler.exp_replay_sampler import ExpReplayMASampler


class BatchMADQN(RLAlgorithm):
    """
    Base class for batch sampling-based deep q network.
    """

    def __init__(self, env, networks, scope=None, n_itr=5000,
                 start_itr=0, batch_size=32, max_path_length=200, discount=0.99,
                 plot=False, pause_for_plot=False, center_adv=True, max_epsilon=1, min_epsilon=0.01,
                 store_paths=False, whole_paths=True, sampler_cls=None,
                 sampler_args=None, force_batch_sampler=True, pre_trained_size=10000, **kwargs):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.ma_mode = kwargs.pop('ma_mode', 'centralized')
        self.env = env
        self.q_network = networks['q_network']
        self.target_q_network = networks['target_q_network']
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.plot = plot
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.force_batch_sampler = force_batch_sampler
        self.loss_flag = True
        self.loss_before = 0
        self.loss_after = 0
        self.mean_kl = 0
        self.max_kl = 0
        self.pre_trained_size = pre_trained_size
        self.total_episodic_rewards = None

        if sampler_cls is None:
            sampler_cls = ExpReplayMASampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(algo=self, **sampler_args)

        if plot:
            from rllab.plotter import plotter
            plotter.init_worker()

        self.init_opt()

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.q_network)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr):
        return self.sampler.obtain_samples(itr)

    def process_samples(self, itr, paths):
        return self.sampler.process_samples(itr, paths)

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.start_worker()

            logger.log("Populating replay memory with random experience...")
            self.sampler.sample_random_paths(self.pre_trained_size, self.ma_mode, self.pre_trained_size)

            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                agent_info = np.zeros((0, 5))
                self.total_episodic_rewards = [[] for _ in range(len(self.env.agents))]
                with logger.prefix('itr #%d | ' % itr):
                    logger.log("Running trajectories...")
                    logger.log("Obtaining samples and optimizing Q network...")
                    p_bar = ProgBarCounter(self.max_path_length)
                    for time_step in range(self.max_path_length):
                        p_bar.inc(time_step+1)
                        paths = self.obtain_samples(itr)
                        samples_data = self.process_samples(itr, paths)
                        self.optimize_policy(itr, samples_data)

                        # For statistics only
                        agent_info = np.vstack([agent_info, samples_data['agent_info']['prob']])

                        if self.sampler.done:
                            break

                    p_bar.stop()
                    self.sampler.done = True
                    self.loss_flag = True
                    self.log_statistics(itr, time_step+1, agent_info)
                    logger.log("Logging statistics...")
                    logger.log("Copying weights to target Q network...")
                    self.target_q_network.set_param_values(self.q_network.get_param_values())
                    logger.log("Logging diagnostics...")
                    self.log_diagnostics(paths)
                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                    if self.store_paths:
                        if isinstance(samples_data, list):
                            params["paths"] = [sd["paths"] for sd in samples_data]
                        else:
                            params["paths"] = samples_data["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - itr_start_time)
                    logger.dump_tabular(with_prefix=False)
                    if self.plot:
                        self.update_plot()
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to " "continue...")
        self.shutdown_worker()

    def log_statistics(self, itr, total_steps, agent_info):

        total_rewards = [sum(reward) for reward in self.total_episodic_rewards]

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageReturn', np.mean(total_rewards))
        logger.record_tabular('NofTrajectories', total_steps)

        ent = np.mean(self.q_network.distribution.entropy({'prob': agent_info}))
        logger.record_tabular('Entropy', ent)

        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(total_rewards))
        logger.record_tabular('MaxReturn', np.max(total_rewards))
        logger.record_tabular('MinReturn', np.min(total_rewards))
        logger.record_tabular("LossBefore", self.loss_before)
        logger.record_tabular("LossAfter", self.loss_after)
        logger.record_tabular('MeanKL', self.mean_kl)
        logger.record_tabular('MaxKL', self.max_kl)

    def log_diagnostics(self, paths):
        if self.ma_mode == 'decentralized':
            import itertools
            self.env.log_diagnostics(list(itertools.chain.from_iterable(paths)))
        else:
            pass

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.q_network, self.max_path_length)
