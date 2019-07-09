import itertools
import math
import sys
import time
from collections import defaultdict

import numpy as np

import agents


class SARSAAgent(agents.BaseAgent):

    def __init__(self, env, num_episodes, **kwargs):
        super(SARSAAgent, self).__init__(env, num_episodes, **kwargs)
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    def _learn(self, state, action, time_step, i_episode):

        # Take a next step and next action
        next_state, R, done, info = self.env.step(action)

        n_step_update = R
        for i in range(self.n_step):

            # Pick the next action
            next_action = self._get_action(next_state)

            if i + 1 == self.n_step or done:

                # Update statistics
                self._update_statistics(n_step_update, time_step, i_episode)

                n_step_update += math.pow(self.discount_factor, i + 1) * self.q_table[next_state][next_action]
            else:
                next_state, R, done, info = self.env.step(next_action)
                n_step_update += math.pow(self.discount_factor, i + 1) * R

        self.q_table[state][action] += self.learning_rate * (n_step_update - self.q_table[state][action])
        return next_state, next_action, done

    def _get_action(self, state):
        return self._get_epsilon_greedy_policy_v0(state)

    def _get_epsilon_greedy_policy_v1(self, state):
        if np.random.uniform(0, 1) < (1 - self.epsilon):
            # take action according to the q function table
            action = np.argmax(self.q_table[state])

        else:
            # take random action: EXPLORE
            action = self.env.action_space.sample()
        return action

    def _get_epsilon_greedy_policy_v0(self, state):
        action_prob = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(self.q_table[state])
        action_prob[best_action] += (1.0 - self.epsilon)
        return np.random.choice(np.arange(len(action_prob)), p=action_prob)

    def accuracy(self):
        raise NotImplementedError

    def train(self):

        super(SARSAAgent, self).train("SARSA", 1.0)

        for i_episode in range(self.num_episodes):

            print("\rRunning Episode {}/{}.".format(i_episode + 1, self.num_episodes), end="")
            sys.stdout.flush()

            # Reset the environment and pick the first action
            state = self.env.reset()
            action = self._get_action(state)

            # One step in the environment
            for time_step in itertools.count():

                # render environment
                if self.render_env:
                    self.env.render()
                    time.sleep(0.5)

                # TD Update
                next_state, next_action, done = self._learn(state, action, time_step, i_episode)

                if done:
                    break

                state = next_state
                action = next_action

        self.exit("Training Completed. Q values published successfully at agent.q_table. "
                  "All evaluation statistics are available at agent.stats")
