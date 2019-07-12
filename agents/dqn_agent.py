import itertools
import math
import sys
import time

import numpy as np
import tensorflow.python.util.deprecation as deprecation
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop

import agents
from lib.utils.function_estimator import ANN
from lib.utils.persistent_solution import Memory
from lib.utils.rl_policy import make_deep_epsilon_greedy_policy

deprecation._PRINT_DEPRECATION_WARNINGS = False


class DQNAgent(agents.BaseAgent):

    def __init__(self, env_name, env, num_episodes, **kwargs):
        super(DQNAgent, self).__init__(env_name, env, num_episodes, **kwargs)
        self.nn = ANN(self.nS, self.nA, self.learning_rate)
        self.nn_target = ANN(self.nS, self.nA, self.learning_rate)
        self.memory = Memory(self.memory_capacity)
        self.policy = make_deep_epsilon_greedy_policy(env)
        self.MAX_EPSILON = 1
        self.MIN_EPSILON = 0.01
        self.epsilon = self.MAX_EPSILON

    def dqnetwork(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.nS, activation='relu'))
        model.add(Dense(self.nA, activation='linear'))
        model.compile(loss='mse', osptimizer=RMSprop(self.learning_rate))
        return model

    def _replay(self):
        self._learn()

    def _learn(self):
        minibatch = self.memory.sample(self.batch_size)

        all_states = np.array([exp[0] for exp in minibatch])
        q = self.nn.predict(all_states)
        all_next_states = np.array([(np.zeros(self.nS) if exp[4] is True else exp[3]) for exp in minibatch])
        next_q = self.nn_target.predict(all_next_states)

        i = 0
        x_train = np.zeros((len(minibatch), self.nS))
        y_train = np.zeros((len(minibatch), self.nA))
        for state, action, reward, next_state, done in minibatch:
            target_q = q[i]
            if done:
                target_q[action] = reward
            else:
                target_q[action] = reward + self.discount_factor * np.amax(next_q[i])

            x_train[i] = state
            y_train[i] = target_q
            i += 1

        self.nn.train(x_train, y_train, batch_size=self.batch_size)

    def _get_action(self, state):
        return self.policy(state, self.nn, self.epsilon)

    def train(self):
        super(DQNAgent, self).train("DQN", 2.0)
        total_time_steps = 1
        for i_episode in range(self.num_episodes):

            print("\rRunning Episode {}/{}".format(i_episode, self.num_episodes), end="")
            sys.stdout.flush()

            # display reward score
            self.score(i_episode)

            # make checkpoint
            if self.make_checkpoint:
                self.save(self.nn, i_episode)

            # Reset the environment and pick the first action
            state = self.env.reset()
            action = self._get_action(state)

            # One step in the environment
            for time_step in itertools.count():

                total_time_steps += time_step

                # render environment
                if self.render_env:
                    self.env.render()
                    time.sleep(0.5)

                # Take a next step and next action
                next_state, R, done, info = self.env.step(action)
                next_action = self._get_action(next_state)

                # Remember experience
                self.memory.add((state, action, R, next_state, done))

                # Replay and train memories
                self._replay()

                # reduce epsilon (because we need less and less exploration)
                # self.epsilon = self.start_epsilon / (1.0 + i_episode * self.decay_rate)
                # self.epsilon *= self.decay_rate
                self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * math.exp(-self.decay_rate * total_time_steps)

                # Update statistics
                self._update_statistics(R, time_step, i_episode)

                if done:
                    # Update Target neural network
                    self.nn_target.set_weights(self.nn.get_weights())
                    break

                action = next_action
                state = next_state

        if self.make_checkpoint:
            self.save(self.nn, self.num_episodes, force_save=True)

        self.exit(self.nn, "DQN agents trained successfully and are available at agent.nn. "
                           "All evaluation statistics are available at agent.stats")
