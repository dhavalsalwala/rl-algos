import random
import sys
import time
from collections import deque

import numpy as np
import tensorflow.python.util.deprecation as deprecation
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

import agents

deprecation._PRINT_DEPRECATION_WARNINGS = False


class DQNAgent(agents.BaseAgent):

    def __init__(self, env_name, env, num_episodes, **kwargs):
        super(DQNAgent, self).__init__(env_name, env, num_episodes, **kwargs)
        self.replay_buffer = deque(maxlen=2000)
        self.nn = []
        for i in range(self.nA):
            self.nn.append(self.dqnetwork())

    def dqnetwork(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.nS, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def _learn(self):

        minibatch = random.sample(self.replay_buffer, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
            else:
                val = []
                for i in self.nn:
                    val.append(i.predict(next_state)[0])
                target = reward + self.discount_factor * np.amax(val)

            self.nn[action].fit(state, [[target]], epochs=1, verbose=0)

        # if self.epsilon > self.epsilon_min:
        #    self.epsilon *= self.epsilon_decay

    def _get_action(self, state):
        return self._get_epsilon_greedy_policy_v1(state)

    def _get_epsilon_greedy_policy_v1(self, state):
        if np.random.uniform(0, 1) < (1 - self.epsilon):
            # take action according to the dq network
            val = []
            for i in self.nn:
                val.append(i.predict(state)[0])
            action = np.argmax(val)
        else:
            # take random action: EXPLORE
            action = self.env.action_space.sample()
        return action

    def train(self):

        super(DQNAgent, self).train("Q Learning", 2.0)
        for i_episode in range(self.num_episodes):

            print("\rRunning Episode {}/{}".format(i_episode + 1, self.num_episodes), end="")
            sys.stdout.flush()

            # reduce epsilon (because we need less and less exploration)
            self.epsilon = self.start_epsilon / (1.0 + i_episode * self.decay_rate)

            # display reward score
            self.score(i_episode)

            # make checkpoint
            if self.make_checkpoint:
                self.save(self.nn, i_episode)

            # Reset the environment and pick the first action
            state = self.env.reset()
            state = np.reshape(state, [1, self.nS])
            action = self._get_action(state)

            # One step in the environment
            for time_step in range(self.MAX_STEPS):

                # render environment
                if self.render_env:
                    self.env.render()
                    time.sleep(0.5)

                # Take a next step and next action
                next_state, R, done, info = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.nS])
                next_action = self._get_action(next_state)

                if done:
                    if time_step + 1 != 200:
                        R = -10
                self.remember(state, action, R, next_state, done)

                # Update statistics
                self._update_statistics(R, time_step, i_episode)

                if done:
                    break

                if len(self.replay_buffer) > self.batch_size:
                    self._learn()

                action = next_action
                state = next_state

        if self.make_checkpoint:
            self.save(self.nn, self.num_episodes, force_save=True)

        self.exit(self.nn, "Q values published successfully at agent.q_table. "
                           "All evaluation statistics are available at agent.stats")
