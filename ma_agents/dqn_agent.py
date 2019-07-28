import tensorflow as tf
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer

from ma_agents.handler.dqn_ma_handler import BatchMADQN


class MADQN(BatchMADQN, Serializable):
    """
    Multi-agent Deep Queue Network.
    """

    def __init__(self, env, optimizer=None, optimizer_args=None, **kwargs):
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
        super(MADQN, self).__init__(env=env, **kwargs)

    @overrides
    def init_opt(self):

        is_recurrent = int(self.policy.recurrent)
        obs_shape = self.env.observation_space.shape
        observations = tf.placeholder(tf.float32, (None, obs_shape[0], obs_shape[1], obs_shape[2]), name='observations')
        next_observations = tf.placeholder(tf.float32, (None, obs_shape[0], obs_shape[1], obs_shape[2]), name='next_observations')
        actions = tf.placeholder(tf.float32, (None, self.env.action_space.n), name='actions')
        done = tf.placeholder(tf.float32, (None, 1), name='done')
        rewards = tf.placeholder(tf.float32, (None, 1), name='rewards')

        self.s_loss = tf.placeholder(tf.float32, name='s_loss')
        self.s_avg_rewards = tf.placeholder(tf.float32, name='s_avg_rewards')
        self.s_total_rewards = tf.placeholder(tf.float32, name='s_total_rewards')

        # target = tf.placeholder(tf.float32, (None, 1), name='target')
        discount_factor = tf.constant(self.discount, name='discount_factor')

        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape),
                              name='old_%s' % k)
            for k, shape in dist.dist_info_specs
        }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
        }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        next_q_value = self.policy.dist_info_sym(next_observations)['prob']
        next_q_value_target_prob = self.target_policy.dist_info_sym(next_observations)['prob']

        # next_q_value__max_indexes = tf.cast(tf.arg_max(next_q_value, 1), dtype=tf.int32)
        # actions_next_flatten = next_q_value__max_indexes + tf.range(0, self.batch_size) * self.env.action_space.n
        # next_q_value_target = tf.gather(tf.reshape(next_q_value_target_prob, [-1]), actions_next_flatten)

        next_q_value_target = \
            tf.reduce_sum(tf.mul(next_q_value_target_prob, tf.one_hot(tf.arg_max(next_q_value, 1), self.env.action_space.n)), reduction_indices=1)

        target_q_value = rewards + discount_factor * next_q_value_target * (1. - done)

        output_prob = self.policy.dist_info_sym(observations)['prob']
        output_q_value = tf.reduce_sum(tf.mul(output_prob, actions), reduction_indices=1)

        loss = tf.reduce_mean(tf.squared_difference(output_q_value, target_q_value))

        dist_info_vars = self.policy.dist_info_sym(observations, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        mean_kl = tf.reduce_mean(kl)
        max_kl = tf.reduce_max(kl)

        input_list = [observations, next_observations, rewards, done, actions] + state_info_vars_list
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
    def optimize_policy(self, time_step, itr, samples_data):
        inputs = ext.extract(samples_data, "observations", "next_observations", "rewards", "done", "actions")
        agent_info = samples_data["agent_info"]
        state_info_list = [agent_info[k] for k in self.policy.state_info_keys]
        inputs += tuple(state_info_list)
        dist_info_list = [agent_info[k] for k in self.policy.distribution.dist_info_keys]

        self.optimizer.optimize(inputs)
        self.loss_after = self.optimizer.loss(inputs)

        self.mean_kl, self.max_kl = self.opt_info['f_kl'](*(list(inputs) + dist_info_list))
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            q_network=self.policy,
            target_qnetwork=self.target_policy,
            env=self.env, )
