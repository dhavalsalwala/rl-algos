import tensorflow as tf
from keras.initializers import Zeros
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam


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

    def train(self, episode_states, episode_actions, discounted_total_rewards):
        loss_, _ = self.session.run([self.loss, self.train_opt], feed_dict={self.state: episode_states,
                                                                            self.actions: episode_actions,
                                                                            self.discounted_total_rewards: discounted_total_rewards
                                                                            })
        return loss_

    def save(self, file_path):
        self.saver.save(self.session, file_path)

    def load(self, file_path):
        self.saver.restore(self.session, file_path)


class PolicyEstimator2:
    def __init__(self, nS, nA, learning_rate=0.001):

        self.nS = nS
        self.nA = nA
        self.learning_rate = learning_rate
        self.model = self.build()

    def build(self):

        model = Sequential()
        model.add(Dense(10, input_dim=self.nS, activation='relu'))
        model.add(Dense(2, input_dim=self.nS, activation='relu'))
        model.add(Dense(2, input_dim=self.nS))
        model.add(Dense(self.nA, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate))
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


class ValueEstimator:
    def __init__(self, nS, nA, learning_rate=0.001):

        self.nS = nS
        self.nA = nA
        self.learning_rate = learning_rate
        self.model = self.build()

    def build(self):

        model = Sequential()
        model.add(Dense(64, input_dim=self.nS, kernel_initializer=Zeros()))
        model.add(Dense(1, activation='linear', kernel_initializer=Zeros()))
        model.compile(loss="mse", optimizer=RMSprop(lr=self.learning_rate))
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
