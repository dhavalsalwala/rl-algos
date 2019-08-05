import random

import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import RMSprop

import utils.common_utils as utils
from utils.persistent import Memory


class ANN:
    def __init__(self, nS, nA, learning_rate=0.001):

        self.nS = nS
        self.nA = nA
        self.learning_rate = learning_rate
        self.model = self.build()

    def build(self):

        model = Sequential()
        model.add(Dense(64, input_dim=self.nS, activation='relu'))
        model.add(Dense(self.nA, activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop(self.learning_rate))
        return model

    def train(self, x, y, batch_size=64, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size, epochs=epochs, verbose=verbose)

    def predict(self, state):
        if state.ndim == 1:
            return self.predict(state.reshape(1, self.nS)).flatten()
        else:
            return self.model.predict(state)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()


class PolicyEstimator:
    """
    Policy Function approximator.
    """

    def __init__(self, nS, nA, session, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None, nS], name="state")
            self.actions = tf.placeholder(tf.int32, [None, nA], name="actions")
            self.discounted_total_rewards = tf.placeholder(tf.float32, [None, ], name="discounted_total_rewards")
            self.session = session

            with tf.name_scope("fc1"):
                fc1 = tf.contrib.layers.fully_connected(inputs=self.state,
                                                        num_outputs=10,
                                                        activation_fn=tf.nn.relu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("fc2"):
                fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                                        num_outputs=nA,
                                                        activation_fn=tf.nn.relu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("fc3"):
                fc3 = tf.contrib.layers.fully_connected(inputs=fc2,
                                                        num_outputs=nA,
                                                        activation_fn=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("softmax"):
                self.action_probs = tf.nn.softmax(fc3)

            with tf.name_scope("loss"):
                neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=self.actions)
                self.loss = tf.reduce_mean(neg_log_prob * self.discounted_total_rewards)

            with tf.name_scope("train"):
                self.train_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss,
                                                                                global_step=tf.contrib.framework.get_global_step())
                self.saver = tf.train.Saver()

    def predict(self, state):
        sess = self.session or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state})

    def train(self, state, episode_actions, discounted_total_rewards):
        if state.ndim == 1:
            state = state.reshape(1, len(state))
        loss_, _ = self.session.run([self.loss, self.train_opt], feed_dict={self.state: state,
                                                                            self.actions: episode_actions,
                                                                            self.discounted_total_rewards: discounted_total_rewards
                                                                            })
        return loss_

    def save(self, file_path):
        self.saver.save(self.session, file_path)

    def load(self, file_path):
        self.saver.restore(self.session, file_path)


class ValueEstimator:
    """
       Value Function approximator.
    """

    def __init__(self, nS, nA, session, learning_rate=0.005, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None, nS], name="state")
            self.target = tf.placeholder(tf.float32, name="target")
            self.session = session

            with tf.name_scope("fc1"):
                fc1 = tf.contrib.layers.fully_connected(inputs=self.state,
                                                        num_outputs=24,
                                                        activation_fn=tf.nn.relu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("fc2"):
                self.fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                                             num_outputs=1,
                                                             activation_fn=tf.keras.activations.linear,
                                                             weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("loss"):
                self.value_estimate = tf.squeeze(self.fc2)
                self.loss = tf.squared_difference(self.value_estimate, self.target)

            with tf.name_scope("train"):
                self.train_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss,
                                                                                global_step=tf.contrib.framework.get_global_step())
                self.saver = tf.train.Saver()

    def predict(self, state):
        if state.ndim == 1:
            state = state.reshape(1, len(state))
        sess = self.session or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state})

    def train(self, state, target):
        if state.ndim == 1:
            state = state.reshape(1, len(state))
        loss_, _ = self.session.run([self.loss, self.train_opt], feed_dict={self.state: state, self.target: target})
        return loss_

    def save(self, file_path):
        self.saver.save(self.session, file_path)

    def load(self, file_path):
        self.saver.restore(self.session, file_path)


class ValueEstimator2:
    """
       Value Function approximator.
    """

    def __init__(self, nS, nA, learning_rate=0.005):

        self.nS = nS
        self.nA = nA
        self.learning_rate = learning_rate
        self.model = self.build()

    def build(self):

        model = Sequential()
        model.add(Dense(24, input_dim=self.nS, kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
        model.compile(loss="mse", optimizer=Adam(lr=0.1))

        return model

    def train(self, x, y, batch_size=64, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size, epochs=epochs, verbose=verbose)

    def predict(self, state):
        if state.ndim == 1:
            state = state.reshape(1, len(state))
        return self.model.predict(state)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()


class ValueFunctionEstimator:
    """
    Critic Value Function Estimator.
    """

    def __init__(self, env, learning_rate=0.0001, value_coefficient=0.5, clip_grads=None, scope="value_estimator"):
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

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=7e-4, decay=0.99, epsilon=1e-5)

            grads = utils.clip_grads(self.optimizer.compute_gradients(self.loss), clip_grads)
            self.train_op = self.optimizer.apply_gradients(grads, global_step=tf.contrib.framework.get_global_step())

    def predict(self, observations, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.observations: observations})

    def update(self, observations, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.observations: observations, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class DQNNetwork:
    def __init__(self, agent_index, env, target_network_update_freq=1000, discount_factor=0.9, batch_size=32, learning_rate=0.001):

        self.env = env
        self.nS = env.observation_space[agent_index].shape[0]
        self.nA = env.action_space[agent_index].n
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_network_update_freq = target_network_update_freq
        self.memory = Memory(100000)
        self.q_estimator = self.build()
        self.target_estimator = self.build()
        self.learning_step = 0
        self.MIN_EPSILON = 0.01
        self.MAX_EPSILON = 1
        self.epsilon = 0.5
        self.decay_rate = 0.001
        self.eps_increment = 1e-5

    def build(self):

        model = Sequential()
        model.add(Dense(64, input_dim=self.nS, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.nA, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))
        return model

    def get_action(self, observation, force_random=False):
        if not force_random and np.random.uniform(0, 1) < self.epsilon:
            # take action according to the dq network
            action = np.argmax(self.predict(observation))
        else:
            # take random action: EXPLORE
            action = random.randrange(self.nA)
        return action

    def replay(self, batch_size=64, epochs=1, verbose=0):

        self.learning_step += 1

        # Update Target neural network
        if self.learning_step % self.target_network_update_freq == 0:
            self.update_target_weights()

        mini_batch = self.memory.sample(self.batch_size)

        all_observation = np.array([exp[0] for exp in mini_batch])
        q = self.q_estimator.predict(all_observation)
        all_next_observation = np.array([(np.zeros(self.nS) if exp[4] is True else exp[3]) for exp in mini_batch])
        next_q = self.q_estimator.predict(all_next_observation)
        target_next_q = self.target_estimator.predict(all_next_observation)

        i = 0
        x_train = np.zeros((len(mini_batch), self.nS))
        y_train = np.zeros((len(mini_batch), self.nA))
        for observation, action, reward, next_observation, done in mini_batch:
            target_q = q[i]
            if done:
                target_q[action] = reward
            else:
                target_q[action] = reward + self.discount_factor * target_next_q[i][np.argmax(next_q[i])]  # Double DQN

            x_train[i] = observation
            y_train[i] = target_q
            i += 1

        history = self.q_estimator.fit(x_train, y_train, batch_size, epochs=epochs, verbose=verbose)
        # self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * math.exp(-self.decay_rate * decay_step)
        if self.epsilon < 0.9:
            self.epsilon += self.eps_increment

        return history

    def add_to_memory(self, observations, actions, R, next_observations, done):
        self.memory.add((observations, actions, R, next_observations, done))

    def update_target_weights(self):
        self.target_estimator.set_weights(self.q_estimator.get_weights())

    def predict(self, observation):
        if observation.ndim == 1:
            observation = observation.reshape(-1, self.nS)
        return self.q_estimator.predict(observation)

    def set_weights(self, weights):
        self.q_estimator.set_weights(weights)

    def get_weights(self):
        return self.q_estimator.get_weights()