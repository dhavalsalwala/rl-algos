import itertools
import sys
import math
import numpy as np
from lib import plotting

from lib.utils.EpsilonGreedyPolicy import make_epsilon_greedy_policy


def apply(n_step, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    model_free_learning algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        n_step: n Step SARSA
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = {key: np.zeros(env.nA) for key in list(range(env.nS))}
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.nA)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
        sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # One step in the environment
        for t in itertools.count():
            # Take a step

            next_state, R, done, info = env.step(action)
            n_step_update = R
            for i in range(n_step):

                # Pick the next action
                next_action_probs = policy(next_state)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

                if i+1 == n_step:

                    # Update statistics
                    stats.episode_rewards[i_episode] += n_step_update
                    stats.episode_lengths[i_episode] = t

                    n_step_update += math.pow(discount_factor, i + 1)*Q[next_state][next_action]
                else:
                    next_state, R, done, info = env.step(next_action)
                    n_step_update += math.pow(discount_factor, i + 1)*R

            # TD Update
            Q[state][action] = Q[state][action] + alpha * (n_step_update - Q[state][action])

            if done:
                break

            action = next_action
            state = next_state

    return Q, stats
