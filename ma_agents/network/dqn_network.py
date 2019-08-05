import random

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from utils.persistent import Memory


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
        self.MIN_EPSILON = 0.01
        self.MAX_EPSILON = 1
        self.epsilon = 0.5
        self.decay_rate = 0.001
        self.eps_increment = 1e-5

    def build(self):

        model = Sequential()
        model.add(Dense(5, input_dim=self.nS, activation='relu'))
        model.add(Dense(5, activation='relu'))
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

    def replay(self, decay_step, batch_size=64, epochs=1, verbose=0):

        # Update Target neural network
        if decay_step % self.target_network_update_freq == 0:
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