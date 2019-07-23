import itertools
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

import agents
from utils.estimator import PolicyEstimator, ValueEstimator
from utils.rl_policy import make_stochastic_policy

deprecation._PRINT_DEPRECATION_WARNINGS = False


class A2CAgent(agents.BaseAgent):

    def __init__(self, env_name, env, num_episodes, session, **kwargs):
        super(A2CAgent, self).__init__(env_name, env, num_episodes, **kwargs)
        self.actor_nn = PolicyEstimator(self.nS, self.nA, session, self.learning_rate)
        self.critic_nn = ValueEstimator(self.nS, self.nA, session)
        self.policy = make_stochastic_policy(self.nA)
        self.session = session

    def _learn(self, state, action, next_state, R, done):

        # Calculate TD Target
        if done:
            td_target = R
        else:
            td_target = R + self.discount_factor * self.critic_nn.predict(next_state)

        # Advantage function (target - value)
        value = self.critic_nn.predict(state)
        td_error_as_advantage = td_target - value

        # Reshaping parameters
        one_hot_action = np.zeros((1, self.nA))
        one_hot_action[0][action] = 1
        state = state.reshape(1, len(state))
        td_error_as_advantage = td_error_as_advantage.reshape((1,))

        # Update the value estimator
        self.critic_nn.train(state, td_target)

        # Update the policy estimator using the td error as advantage estimate
        self.actor_nn.train(state, one_hot_action, td_error_as_advantage)

    def _get_action(self, state):
        return self.policy(state, self.actor_nn)

    def train(self):
        self.session.run(tf.global_variables_initializer())
        super(A2CAgent, self).train("REINFORCE", 2.0)
        for i_episode in range(self.num_episodes):

            print("\rRunning Episode {}/{}".format(i_episode, self.num_episodes), end="")
            sys.stdout.flush()

            # display reward score
            self.score(i_episode)

            # make checkpoint
            if self.make_checkpoint:
                self.save(i_episode, is_tensor=True, is_a2c=True)

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

                # learn experience
                self._learn(state, action, next_state, R, done)

                # Update statistics
                self._update_statistics(R, time_step, i_episode)

                if done:
                    break

                action = next_action
                state = next_state

        if self.make_checkpoint:
            self.save(self.num_episodes, is_tensor=True, force_save=True, is_a2c=True)

        self.exit("A2C agents trained successfully and are available at agent.agent_nn. "
                  "All evaluation statistics are available at agent.stats")
