import tensorflow as tf
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer

from ma_agents.handler.a2c_ma_handler import BatchMAA2C


class MAA2C(BatchMAA2C, Serializable):
    """
    Multi-agent Deep Queue Network.
    """

    def __init__(self, env, actor_learning_rate=1e-4, critic_learning_rate=1e-3, optimizer=None, optimizer_args=None, **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1, )
            optimizer_args = dict(default_args, **{'learning_rate': critic_learning_rate})
            self.critic_optimizer = FirstOrderOptimizer(**optimizer_args)
            optimizer_args = dict(default_args, **{'learning_rate': actor_learning_rate})
            self.actor_optimizer = FirstOrderOptimizer(**optimizer_args)
        self.opt_info = None
        super(MAA2C, self).__init__(env=env, **kwargs)

    @overrides
    def init_opt(self):

        is_recurrent = int(self.policy.recurrent)
        obs_shape = self.env.observation_space.shape
        observations = tf.placeholder(tf.float32, (None, obs_shape[0], obs_shape[1], obs_shape[2]), name='observations')
        actions = self.env.action_space.new_tensor_variable('actions', extra_dims=1 + is_recurrent,)
        td_target = tensor_utils.new_tensor(name='td_target', ndim=1 + is_recurrent, dtype=tf.float32,)
        td_error_advantage = tensor_utils.new_tensor(name='td_error_advantage', ndim=1 + is_recurrent, dtype=tf.float32,)

        self.s_loss = tf.placeholder(tf.float32, name='s_loss')
        self.s_avg_rewards = tf.placeholder(tf.float32, name='s_avg_rewards')
        self.s_total_rewards = tf.placeholder(tf.float32, name='s_total_rewards')

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
        logli = dist.log_likelihood_sym(actions, dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        actor_loss = -tf.reduce_mean(logli * td_error_advantage)
        mean_kl = tf.reduce_mean(kl)
        max_kl = tf.reduce_max(kl)

        input_list_actor = [observations, actions, td_error_advantage] + state_info_vars_list
        self.actor_optimizer.update_opt(loss=actor_loss, target=self.policy, inputs=input_list_actor)

        value_estimates = tf.squeeze(self.critic.dist_info_sym(observations)['prob'], squeeze_dims=[1, ])
        critic_loss = tf.reduce_mean(tf.squared_difference(value_estimates, td_target))
        input_list_critic = [observations, td_target]
        self.critic_optimizer.update_opt(loss=critic_loss, target=self.critic, inputs=input_list_critic)

        f_kl = tensor_utils.compile_function(
            inputs=input_list_actor + old_dist_info_vars_list,
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

        actor_inputs = ext.extract(samples_data, "observations", "actions", "td_error")
        agent_info = samples_data["agent_info"]
        state_info_list = [agent_info[k] for k in self.policy.state_info_keys]
        actor_inputs += tuple(state_info_list)
        dist_info_list = [agent_info[k] for k in self.policy.distribution.dist_info_keys]

        self.actor_optimizer.optimize(actor_inputs)
        self.loss_after = self.actor_optimizer.loss(actor_inputs)

        critic_inputs = ext.extract(samples_data, "observations", "td_target")
        self.critic_optimizer.optimize(critic_inputs)

        self.mean_kl, self.max_kl = self.opt_info['f_kl'](*(list(actor_inputs) + dist_info_list))
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            actor_network=self.policy,
            critic_network=self.critic,
            env=self.env, )
