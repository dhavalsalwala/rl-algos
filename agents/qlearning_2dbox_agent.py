import sys
import time

import numpy as np

import agents


class QLearningAgent(agents.BaseAgent):

    def __init__(self, env_name, env, num_episodes, **kwargs):
        super(QLearningAgent, self).__init__(env_name, env, num_episodes, **kwargs)
        self.q_table = np.random.uniform(low=-1, high=1, size=(self.nS[0], self.nS[1], self.nA))
        self.update_counts = np.zeros((self.nS[0], self.nS[1], self.nA), dtype=np.dtype(int))

    def _learn(self, state, action, next_state, next_action, R):

        # reduce learning rate based on state visits
        self.learning_rate = self.start_learning_rate / (1.0 + self.update_counts[state[0]][state[1]][action] * self.decay_rate)
        self.update_counts[state[0]][state[1]][action] += 1

        next_q = np.max(self.q_table[next_state[0]][next_state[1]])
        self.q_table[state[0]][state[1]][action] += self.learning_rate * (R + self.discount_factor * next_q -
                                                                          self.q_table[state[0]][state[1]][action])

    def _get_action(self, state):
        return self._get_epsilon_greedy_policy_v1(state)

    def _get_epsilon_greedy_policy_v1(self, state):
        if np.random.uniform(0, 1) < (1 - self.epsilon):
            # take random action: EXPLORE
            action = self.env.action_space.sample()
        else:
            # take action according to the q function table
            action = np.argmax(self.q_table[state[0]][state[1]])
        return action

    def _get_epsilon_greedy_policy_v0(self, state):
        action_prob = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(self.q_table[state[0]][state[1]])
        action_prob[best_action] += (1.0 - self.epsilon)
        return np.random.choice(np.arange(len(action_prob)), p=action_prob)

    def train(self):

        super(QLearningAgent, self).train("Q Learning", 2.0)
        for i_episode in range(self.num_episodes):

            print("\rRunning Episode {}/{}".format(i_episode + 1, self.num_episodes), end="")
            sys.stdout.flush()

            # reduce epsilon (because we need less and less exploration)
            self.epsilon = self.start_epsilon / (1.0 + i_episode * self.decay_rate)

            # display reward score
            self.score(i_episode)

            # make checkpoint
            if self.make_checkpoint:
                self.save(self.q_table, i_episode)

            # Reset the environment and pick the first action
            state = self.env.reset()
            state_adj = (state - self.env.observation_space.low) * np.array([10, 100])
            state_adj = np.round(state_adj, 0).astype(int)

            action = self._get_action(state_adj)

            # One step in the environment
            for time_step in range(self.MAX_STEPS):

                # render environment
                if self.render_env:
                    self.env.render()
                    time.sleep(0.5)

                # Take a next step and next action
                next_state, R, done, info = self.env.step(action)
                next_state_adj = (next_state - self.env.observation_space.low) * np.array([10, 100])
                next_state_adj = np.round(next_state_adj, 0).astype(int)

                next_action = self._get_action(next_state_adj)

                # Update statistics
                self._update_statistics(R, time_step, i_episode)

                # TD Update
                self._learn(state_adj, action, next_state_adj, next_action, R)

                if done:
                    break

                action = next_action
                state_adj = next_state_adj

        if self.make_checkpoint:
            self.save(self.q_table, self.num_episodes, force_save=True)

        self.exit(self.q_table, "Training Completed. Q values published successfully at agent.q_table. "
                  "All evaluation statistics are available at agent.stats")
