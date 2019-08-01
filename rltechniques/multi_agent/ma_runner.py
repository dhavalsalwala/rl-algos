import os.path as osp

import rllab.misc.logger as logger
from rllab import config
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.misc.ext import set_seed
from rllab.sampler import parallel_sampler
from rllabwrapper import RLLabEnv
from sandbox.rocky.tf.core.network import ConvNetwork
from sandbox.rocky.tf.envs.base import MATfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.spaces.discrete import Discrete

from ma_agents.a2c_agent import MAA2C
from ma_agents.dqn_agent import MADQN
from ma_agents.reinforce_agent import MAReinforce
from . import *


class Runner(object):

    def __init__(self, env, args):

        self.env = env
        self.args = args

        # Parallel setup
        parallel_sampler.initialize(n_parallel=args.n_parallel)
        if args.seed is not None:
            set_seed(args.seed)
            parallel_sampler.set_seed(args.seed)

        index = 1
        env, policy = self.parse_env_args(env, args)
        self.algo = self.setup(env, policy, start_itr=index)

    def start_training(self):
        self.algo.train()

    def parse_env_args(self, env, args):

        if isinstance(args, dict):
            args = tonamedtuple(args)

        env = RLLabEnv(env, ma_mode=args.control)
        # Multi-agent wrapper
        env = MATfEnv(env)

        # Policy
        if args.conv:
            strides = tuple(args.conv_strides)
            chans = tuple(args.conv_channels)
            filts = tuple(args.conv_filters)

            assert len(strides) == len(chans) == len(filts), "strides, chans and filts not equal"
            # only discrete actions supported, should be straightforward to extend to continuous
            assert isinstance(env.spec.action_space,
                              Discrete), "Only discrete action spaces support conv"
            feature_network = ConvNetwork(name='feature_net',
                                          input_shape=env.spec.observation_space.shape,
                                          output_dim=env.spec.action_space.n, conv_filters=chans,
                                          conv_filter_sizes=filts, conv_strides=strides,
                                          conv_pads=(args.conv_pads,) * len(chans),
                                          hidden_sizes=tuple(args.policy_hidden),
                                          hidden_nonlinearity=tf.nn.relu,
                                          output_nonlinearity=tf.nn.softmax,
                                          batch_normalization=args.batch_normalization)
            if args.algo == 'dqn':
                q_network = CategoricalMLPPolicy(name='q_network', env_spec=env.spec,
                                                 prob_network=feature_network)
                target_q_network = CategoricalMLPPolicy(name='target_q_network', env_spec=env.spec,
                                                        prob_network=feature_network)
                policy = {'q_network': q_network, 'target_q_network': target_q_network}

            else:
                policy = CategoricalMLPPolicy(name='policy', env_spec=env.spec,
                                              prob_network=feature_network)
        else:
            if isinstance(env.spec.action_space, Box):
                policy = GaussianMLPPolicy(env_spec=env.spec,
                                           hidden_sizes=tuple(args.policy_hidden),
                                           min_std=args.min_std, name='policy')
            elif isinstance(env.spec.action_space, Discrete):
                policy = CategoricalMLPPolicy(env_spec=env.spec,
                                              hidden_sizes=tuple(args.policy_hidden), name='policy')
            else:
                raise NotImplementedError(env.spec.action_space)

        return env, policy

    def setup(self, env, policy, start_itr):

        # Baseline
        if self.args.baseline_type == 'linear':
            baseline = LinearFeatureBaseline(env_spec=env.spec)
        elif self.args.baseline_type == 'zero':
            baseline = ZeroBaseline(env_spec=env.spec)
        else:
            raise NotImplementedError(self.args.baseline_type)

        # Logger
        default_log_dir = config.LOG_DIR
        if self.args.log_dir is None:
            log_dir = osp.join(default_log_dir, self.args.exp_name)
        else:
            log_dir = self.args.log_dir

        tabular_log_file = osp.join(log_dir, self.args.tabular_log_file)
        text_log_file = osp.join(log_dir, self.args.text_log_file)
        params_log_file = osp.join(log_dir, self.args.params_log_file)

        logger.log_parameters_lite(params_log_file, self.args)
        logger.add_text_output(text_log_file)
        logger.add_tabular_output(tabular_log_file)
        # prev_snapshot_dir = logger.get_snapshot_dir()
        # prev_mode = logger.get_snapshot_mode()
        logger.set_snapshot_dir(log_dir)
        logger.set_snapshot_mode(self.args.snapshot_mode)
        logger.set_log_tabular_only(self.args.log_tabular_only)
        logger.push_prefix("[%s] " % self.args.exp_name)

        if self.args.algo == 'reinforce':
            algo = MAReinforce(env=env, policy_or_policies=policy, plot=False, baseline_or_baselines=baseline,
                               batch_size=self.args.batch_size, pause_for_plot=True, start_itr=start_itr,
                               max_path_length=self.args.max_path_length, n_itr=self.args.n_iter,
                               discount=self.args.discount, gae_lambda=self.args.gae_lambda,
                               step_size=self.args.step_size, ma_mode=self.args.control, save_param_update=self.args.save_param_update)

        elif self.args.algo == 'dqn':
            algo = MADQN(env=env, networks=policy, plot=False,
                         batch_size=self.args.batch_size, pause_for_plot=True, start_itr=start_itr,
                         max_path_length=self.args.max_path_length, n_itr=self.args.n_iter,
                         discount=self.args.discount, ma_mode=self.args.control,
                         pre_trained_size=self.args.replay_pre_trained_size, target_network_update=self.args.target_network_update,
                         save_param_update=self.args.save_param_update)

        elif self.args.algo == 'a2c':
            algo = MAA2C(env=env, policy_or_policies=policy, plot=False, baseline_or_baselines=baseline,
                         batch_size=self.args.batch_size, pause_for_plot=True, start_itr=start_itr,
                         max_path_length=self.args.max_path_length, n_itr=self.args.n_iter,
                         discount=self.args.discount, ma_mode=self.args.control,
                         actor_learning_rate=self.args.policy_lr, critic_learning_rate=self.args.qfunc_lr,
                         value_coefficient=0.5, entropy_coefficient=0.01)

        return algo
