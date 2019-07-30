import tensorflow as tf
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer

from ma_agents.handler.reinforce_ma_handler import ReinforceMABase


class MAReinforce(ReinforceMABase, Serializable):
    """
    Multi-agent Vanilla Policy Gradient.
    """

    def __init__(self, env, policy_or_policies, baseline_or_baselines, optimizer=None,
                 optimizer_args=None, **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1, )
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
            optimizer = FirstOrderOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.opt_info = None
        super(MAReinforce, self).__init__(env=env, policy_or_policies=policy_or_policies,
                                          baseline_or_baselines=baseline_or_baselines, **kwargs)

    @overrides
    def init_opt(self):

        observations = self.env.observation_space.new_tensor_variable('observations', extra_dims=1, )
        actions = self.env.action_space.new_tensor_variable('action', extra_dims=1, )
        advantage = tensor_utils.new_tensor(name='advantage', ndim=1, dtype=tf.float32, )
        dist = self.policy.distribution

        self.s_loss = tf.placeholder(tf.float32, name='s_loss')
        self.s_avg_rewards = tf.placeholder(tf.float32, name='s_avg_rewards')
        self.s_total_rewards = tf.placeholder(tf.float32, name='s_total_rewards')

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * 1 + list(shape),
                              name='old_%s' % k)
            for k, shape in dist.dist_info_specs
        }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * 1 + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
        }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        dist_info_vars = self.policy.dist_info_sym(observations, state_info_vars)
        logli = dist.log_likelihood_sym(actions, dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        loss = -tf.reduce_mean(logli * advantage)
        mean_kl = tf.reduce_mean(kl)
        max_kl = tf.reduce_max(kl)

        input_list = [observations, actions, advantage] + state_info_vars_list

        self.optimizer.update_opt(loss=loss, target=self.policy, inputs=input_list)

        f_kl = tensor_utils.compile_function(
            inputs=input_list + old_dist_info_vars_list,
            outputs=[mean_kl, max_kl], )
        self.opt_info = dict(f_kl=f_kl, )

        self.writer = tf.train.SummaryWriter("summary/")
        self.write_op = tf.merge_summary([
            tf.scalar_summary("Loss", self.s_loss),
            tf.scalar_summary("Total Rewards", self.s_total_rewards),
            tf.scalar_summary("Avg Rewards", self.s_avg_rewards)
        ])

    @overrides
    def optimize_policy(self, itr, samples_data):
        inputs = ext.extract(samples_data, "observations", "actions", "advantages")
        agent_info = samples_data["agent_infos"]
        state_info_list = [agent_info[k] for k in self.policy.state_info_keys]
        inputs += tuple(state_info_list)
        dist_info_list = [agent_info[k] for k in self.policy.distribution.dist_info_keys]
        loss_before = self.optimizer.loss(inputs)
        self.optimizer.optimize(inputs)
        self.loss = self.optimizer.loss(inputs)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", self.loss)

        mean_kl, max_kl = self.opt_info['f_kl'](*(list(inputs) + dist_info_list))
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('MaxKL', max_kl)

        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env, )
