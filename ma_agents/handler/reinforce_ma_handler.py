import time
from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
import rllab.plotter as plotter
from sandbox.rocky.tf.policies.base import Policy
import tensorflow as tf
from ma_agents.sampler.reinforce_sampler import ReinforceMASampler


class ReinforceMABase(RLAlgorithm):

    def __init__(self, env, policy_or_policies, baseline_or_baselines, scope=None, n_itr=500,
                 start_itr=0, batch_size=5000, max_path_length=500, discount=0.99, gae_lambda=1,
                 plot=False, pause_for_plot=False, center_adv=True, positive_adv=False,
                 store_paths=False, whole_paths=True, fixed_horizon=False, sampler_cls=None,
                 sampler_args=None, save_param_update=125, force_batch_sampler=True, **kwargs):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.ma_mode = kwargs.pop('ma_mode', 'centralized')
        self.env = env
        self.writer = None
        self.write_op = None
        self.policy = policy_or_policies
        self.baseline = baseline_or_baselines
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.force_batch_sampler = force_batch_sampler
        self.save_param_update = save_param_update
        self.loss = 0
        self.avg_rewards = 0
        self.total_rewards = 0
        self.s_loss = None
        self.s_avg_rewards = None
        self.s_total_rewards = None

        if sampler_cls is None:
            sampler_cls = ReinforceMASampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        self.init_opt()

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

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
            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):
                    logger.log("Obtaining samples...")
                    paths = self.obtain_samples(itr)
                    logger.log("Processing samples...")
                    # TODO Process appropriately for concurrent or decentralized
                    samples_data = self.process_samples(itr, paths)
                    logger.log("Logging diagnostics...")
                    self.log_diagnostics(paths)
                    logger.log("Optimizing policy...")
                    self.optimize_policy(itr, samples_data)

                    if itr % self.save_param_update == 0:
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
                    self.log_summary(itr)
                    if self.plot:
                        self.update_plot()
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to " "continue...")
        self.shutdown_worker()

    def log_summary(self, itr):
        # Write TF Summaries
        sess = tf.get_default_session()
        summary = sess.run(self.write_op, feed_dict={self.s_loss: self.loss,
                                                     self.s_avg_rewards: self.avg_rewards,
                                                     self.s_total_rewards: self.total_rewards})
        self.writer.add_summary(summary, itr)
        self.writer.flush()

    def log_diagnostics(self, paths):
        import itertools
        self.env.log_diagnostics(list(itertools.chain.from_iterable(paths)))

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
            plotter.update_plot(self.policy, self.max_path_length)
