import itertools
import sys

import numpy as np
from lib import plotting

from lib.utils.EpsilonGreedyPolicy import make_epsilon_greedy_policy


def apply(env, num_episodes, discount_factor=0.01, alpha=0.01, epsilon=0.9):
    """
    model_free_learning algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
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
    # Q = defaultdict(lambda: np.zeros(env.action_space.n))
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

            # Pick the next action
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            # Update statistics
            stats.episode_rewards[i_episode] += R
            stats.episode_lengths[i_episode] = t

            # TD Update
            Q[state][action] = Q[state][action] + alpha * (R + discount_factor *
                                                           Q[next_state][np.argmax(Q[next_state])] - Q[state][action])

            if done:
                break

            action = next_action
            state = next_state

    return Q, stats
