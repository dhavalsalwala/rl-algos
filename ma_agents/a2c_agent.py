import tensorflow as tf
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
import numpy as np
from ma_agents.handler.a2c_ma_handler import A2CMABase


class MAA2C(A2CMABase, Serializable):
    """
    Multi-agent Advantage Actor Critic.
    """

    def __init__(self, env, actor_learning_rate=1e-4, optimizer=None, optimizer_args=None, **kwargs):
        Serializable.quick_init(self, locals())
        super(MAA2C, self).__init__(env=env, **kwargs)
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
                tf_optimizer_cls=tf.train.RMSPropOptimizer,
                tf_optimizer_args=dict(learning_rate=7e-4, decay=0.99, epsilon=1e-5)
            )
            optimizer_args = dict(default_args, **{'clip_grads': self.clip_grads})
            self.optimizer = FirstOrderOptimizer(**optimizer_args)
        self.opt_info = None
        self.init_opt()

    @overrides
    def init_opt(self):

        is_recurrent = int(self.policy.recurrent)
        obs_shape = self.env.observation_space.shape
        observations = tf.placeholder(tf.float32, (None, obs_shape[0], obs_shape[1], obs_shape[2]), name='observations')
        actions = tf.placeholder(tf.float32, (None, self.env.action_space.n), name='observations')
        td_error_advantage = tensor_utils.new_tensor(name='td_error_advantage', ndim=1 + is_recurrent, dtype=tf.float32,)
        entropy_coefficient = tf.constant(self.entropy_coefficient, name='entropy_coefficient')

        self.actor_loss = tf.placeholder(tf.float32, name='actor_loss')
        self.critic_loss = tf.placeholder(tf.float32, name='critic_loss')
        self.entropy_loss = tf.placeholder(tf.float32, name='entropy_loss')
        self.avg_rewards = tf.placeholder(tf.float32, name='avg_rewards')
        self.total_rewards = tf.placeholder(tf.float32, name='total_rewards')

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

        dist_info_vars = self.policy.dist_info_sym(observations, state_info_vars)

        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        mean_kl = tf.reduce_mean(kl)
        max_kl = tf.reduce_max(kl)

        # log_likelihood_prob = dist.log_likelihood_sym(actions, dist_info_vars)
        # actor_loss = -tf.reduce_mean(log_likelihood_prob * td_error_advantage)

        neg_log_policy = -tf.log(tf.clip_by_value(dist_info_vars['prob'], 1e-7, 1))
        actor_loss = tf.reduce_mean(tf.reduce_sum(neg_log_policy * actions, reduction_indices=1) * td_error_advantage)
        entropy_loss = tf.reduce_mean(tf.reduce_sum(dist_info_vars['prob'] * neg_log_policy, reduction_indices=1))
        # entropy_loss = tf.reduce_mean(self.policy.distribution.entropy_sym(dist_info_vars))

        total_loss = actor_loss - (entropy_loss * entropy_coefficient)
        input_list_actor = [observations, actions, td_error_advantage] + state_info_vars_list
        self.optimizer.update_opt(loss=total_loss, target=self.policy, inputs=input_list_actor)

        f_kl = tensor_utils.compile_function(
            inputs=input_list_actor + old_dist_info_vars_list,
            outputs=[mean_kl, max_kl], )
        self.opt_info = dict(f_kl=f_kl, )

        self.writer = tf.train.SummaryWriter("summary/")
        self.write_op = tf.merge_summary([
            tf.scalar_summary("Actor Loss", self.actor_loss),
            tf.scalar_summary("Critic Loss", self.critic_loss),
            tf.scalar_summary("Entropy", self.entropy_loss),
            tf.scalar_summary("Total Rewards", self.total_rewards),
            tf.scalar_summary("Avg Rewards", self.avg_rewards)
        ])

    @overrides
    def optimize_policy(self, itr, samples_data):

        inputs = ext.extract(samples_data, "observations", "actions", "td_error")
        agent_info = samples_data["agent_info"]
        state_info_list = [agent_info[k] for k in self.policy.state_info_keys]
        inputs += tuple(state_info_list)
        dist_info_list = [agent_info[k] for k in self.policy.distribution.dist_info_keys]

        loss_before = self.optimizer.loss(inputs)
        self.optimizer.optimize(inputs)
        loss_after = self.optimizer.loss(inputs)

        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)

        rewards = samples_data['rewards']
        entropy_loss = np.mean(self.policy.distribution.entropy(agent_info))
        self.log_summary(itr, loss_after, self.value_loss, entropy_loss, np.mean(rewards), np.sum(rewards))

        mean_kl, max_kl = self.opt_info['f_kl'](*(list(inputs) + dist_info_list))
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('MaxKL', max_kl)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            actor_network=self.policy,
            critic_network=self.value_estimator,
            env=self.env, )
