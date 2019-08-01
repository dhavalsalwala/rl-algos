import tensorflow as tf


class ValueEstimator:
    """
    Value Function Estimator.
    """

    def __init__(self, env, learning_rate=0.0001, value_coefficient=0.5, scope="value_estimator"):
        with tf.variable_scope(scope):
            obs_shape = env.observation_space.shape
            self.observations = tf.placeholder(tf.float32, (None, obs_shape[0], obs_shape[1], obs_shape[2]), name='observations')
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            conv1 = tf.contrib.layers.conv2d(inputs=self.observations, num_outputs=32, kernel_size=[3, 3], stride=(1, 1), activation_fn=tf.nn.relu,
                                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=[3, 3], stride=(1, 1), activation_fn=tf.nn.relu,
                                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=64, kernel_size=[3, 3], stride=(1, 1), activation_fn=tf.nn.relu,
                                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv_out = tf.contrib.layers.flatten(conv3)
            dense = tf.contrib.layers.fully_connected(conv_out, 512, activation_fn=tf.nn.relu)
            value_function = tf.contrib.layers.fully_connected(dense, 1, activation_fn=None)

            self.value_estimate = tf.squeeze(value_function)
            self.loss = value_coefficient * tf.reduce_mean(tf.squared_difference(self.value_estimate, self.target))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, observations, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.observations: observations})

    def update(self, observations, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.observations: observations, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss
