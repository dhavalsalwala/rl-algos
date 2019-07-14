import itertools
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

import agents
from utils import PGMemory
from utils import PolicyEstimator, ValueEstimator
from utils import make_stochastic_policy

deprecation._PRINT_DEPRECATION_WARNINGS = False


class REINFORCEAgent(agents.BaseAgent):

    def __init__(self, env_name, env, num_episodes, session, **kwargs):
        super(REINFORCEAgent, self).__init__(env_name, env, num_episodes, **kwargs)
        self.policy_nn = PolicyEstimator(self.nS, self.nA, session, self.learning_rate)
        self.value_nn = ValueEstimator(self.nS, self.nA, self.learning_rate)
        self.memory = PGMemory()
        self.policy = make_stochastic_policy(self.nA)
        self.session = session

    def discount_and_normalize_rewards(self, episode_rewards):
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * self.discount_factor + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative

        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        discounted_episode_rewards = (discounted_episode_rewards - mean) / std

        return discounted_episode_rewards

    def _learn(self):
        episode_length = len(self.memory.buffer)
        total_return = self.discount_and_normalize_rewards(np.array([t.reward for i, t in enumerate(self.memory.buffer[:])]))

        x_train_value = np.zeros((episode_length, self.nS))
        y_train_value = np.zeros(episode_length)

        episode_states = np.zeros((episode_length, self.nS))
        episode_actions = np.zeros((episode_length, self.nA))

        for t, transition in enumerate(self.memory.buffer):
            # total_return = sum(self.discount_factor ** i * t.reward for i, t in enumerate(self.memory.buffer[t:]))

            # Calculate baseline/advantage
            baseline_value = self.value_nn.predict(transition.state)
            advantage = total_return - baseline_value

            x_train_value[t] = transition.state
            y_train_value[t] = total_return[t]

            episode_states[t] = transition.state
            episode_actions[t][transition.action] = 1

        # Update our value estimator
        # self.value_nn.train(x_train_value, y_train_value)
        # Update our policy estimator
        self.policy_nn.train(episode_states, episode_actions, total_return)

        self.memory.clear()

    def _get_action(self, state):
        return self.policy(state, self.policy_nn)

    def train(self):
        self.session.run(tf.global_variables_initializer())
        super(REINFORCEAgent, self).train("REINFORCE", 2.0)
        for i_episode in range(self.num_episodes):

            print("\rRunning Episode {}/{}".format(i_episode, self.num_episodes), end="")
            sys.stdout.flush()

            # display reward score
            self.score(i_episode)

            # make checkpoint
            if self.make_checkpoint:
                self.save(i_episode, is_tensor=True)

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

                # Remember experience
                self.memory.add(state=state, action=action, reward=R, next_state=next_state, done=done)

                # Update statistics
                self._update_statistics(R, time_step, i_episode)

                if done:
                    # Go through the episode and make policy updates
                    self._learn()
                    break

                action = next_action
                state = next_state

        if self.make_checkpoint:
            self.save(self.num_episodes, is_tensor=True, force_save=True)

        self.exit("REINFORCE agents trained successfully and are available at agent.policy_nn. "
                  "All evaluation statistics are available at agent.stats")
