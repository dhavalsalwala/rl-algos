import tensorflow as tf
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc.overrides import overrides
from ma_agents.handler.dqn_ma_handler import BatchMADQN
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer


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

        is_recurrent = int(self.q_network.recurrent)
        obs_shape = self.env.observation_space.shape
        observations = tf.placeholder(tf.float32, (None, obs_shape[0], obs_shape[1], obs_shape[2]), name='observations')
        next_observations = tf.placeholder(tf.float32, (None, obs_shape[0], obs_shape[1], obs_shape[2]), name='next_observations')
        actions = tf.placeholder(tf.float32, (None, 1, self.env.action_space.n), name='actions')
        done = tf.placeholder(tf.float32, (None, 1), name='done')
        rewards = tf.placeholder(tf.float32, (None, 1), name='rewards')
        discount_factor = tf.constant(self.discount, name='discount_factor')

        dist = self.q_network.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape),
                              name='old_%s' % k)
            for k, shape in dist.dist_info_specs
        }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.q_network.state_info_specs
        }
        state_info_vars_list = [state_info_vars[k] for k in self.q_network.state_info_keys]

        next_q_value = self.target_q_network.dist_info_sym(next_observations)['prob']
        next_q_value = tf.reduce_max(next_q_value) * done
        target_q_value = rewards + discount_factor * next_q_value

        output_q_value = tf.reduce_sum(tf.mul(self.q_network.dist_info_sym(observations)['prob'], actions), reduction_indices=[1, ])
        loss = tf.reduce_mean(tf.squared_difference(target_q_value, output_q_value))

        dist_info_vars = self.q_network.dist_info_sym(observations, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        mean_kl = tf.reduce_mean(kl)
        max_kl = tf.reduce_max(kl)

        input_list = [observations, next_observations, actions, done, rewards] + state_info_vars_list
        self.optimizer.update_opt(loss=loss, target=self.q_network, inputs=input_list)

        f_kl = tensor_utils.compile_function(
            inputs=input_list + old_dist_info_vars_list,
            outputs=[mean_kl, max_kl],)
        self.opt_info = dict(f_kl=f_kl,)

    @overrides
    def optimize_policy(self, itr, samples_data):
        inputs = ext.extract(samples_data, "observations", "next_observations", "actions", "done", "rewards")
        agent_info = samples_data["agent_info"]
        state_info_list = [agent_info[k] for k in self.q_network.state_info_keys]
        inputs += tuple(state_info_list)
        dist_info_list = [agent_info[k] for k in self.q_network.distribution.dist_info_keys]

        self.loss_before = self.optimizer.loss(inputs)
        self.optimizer.optimize(inputs)
        self.loss_after = self.optimizer.loss(inputs)

        self.mean_kl, self.max_kl = self.opt_info['f_kl'](*(list(inputs) + dist_info_list))
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            q_network=self.q_network,
            target_qnetwork=self.target_q_network,
            env=self.env, )
