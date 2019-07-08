import agents
import numpy as np
import sys
import itertools
import time
from collections import defaultdict


class SARSAAgent(agents.BaseAgent):

    def __init__(self, env, num_episodes, **kwargs):
        super(SARSAAgent, self).__init__(env, num_episodes, **kwargs)
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    def _learn(self, state, action, next_state, next_action, R):
        old_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        new_q = old_q + self.learning_rate * (R + self.discount_factor * next_q - old_q)
        self.q_table[state][action] = new_q

    def _get_action(self, state):
        return self._get_epsilon_greedy_policy_v0(state)

    def _get_epsilon_greedy_policy_v1(self, state):
        if np.random.rand() < self.epsilon:
            # take random action
            action = np.random.choice(self.nA)
        else:
            # take action according to the q function table
            state_action = self.q_table[state]
            action = np.argmax(state_action)
        return action

    def _get_epsilon_greedy_policy_v0(self, state):
        action_prob = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(self.q_table[state])
        action_prob[best_action] += (1.0 - self.epsilon)
        return np.random.choice(np.arange(len(action_prob)), p=action_prob)

    def accuracy(self):
        raise NotImplementedError

    def _update_statistics(self, R, time_step, i_episode):
        self.stats.episode_rewards[i_episode] += R
        self.stats.episode_lengths[i_episode] = time_step

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

                # Take a next step and next action
                next_state, R, done, info = self.env.step(action)
                next_action = self._get_action(next_state)

                # Update statistics
                self._update_statistics(R, time_step, i_episode)

                # TD Update
                self._learn(state, action, next_state, next_action, R)

                if done:
                    break

                action = next_action
                state = next_state

        self.exit("Training Completed. Q values published successfully at agent.q_table. "
                  "All evaluation statistics are available at agent.stats")
