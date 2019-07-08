import itertools
import sys
import matplotlib.pyplot as plt
import numpy as np
from lib import plotting

from lib.utils.EpsilonGreedyPolicy import make_epsilon_greedy_policy


def apply(env, num_episodes, discount_factor=1, alpha=0.005, epsilon=0.05):
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

    n_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
    n_states = np.round(n_states, 0).astype(int) + 1

    Q = np.random.uniform(low=-1, high=1, size=(n_states[0], n_states[1], env.action_space.n))
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n, flag=True)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
        sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        state_adj = (state - env.observation_space.low) * np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)

        action_probs = policy(state_adj)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # One step in the environment
        for t in itertools.count():

            # Take a step
            next_state, R, done, info = env.step(action)

            next_state_adj = (next_state - env.observation_space.low) * np.array([10, 100])
            next_state_adj = np.round(next_state_adj, 0).astype(int)

            if next_state_adj[0] >= 18:
                print("hurray: "+str(i_episode))

            # Pick the next action
            next_action_probs = policy(next_state_adj)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            # Update statistics
            stats.episode_rewards[i_episode] += R
            stats.episode_lengths[i_episode] = t

            # TD Update
            Q[state_adj[0]][state_adj[1]][action] = Q[state_adj[0]][state_adj[1]][action] + alpha * \
                                            (R + discount_factor * Q[next_state_adj[0]][next_state_adj[1]]
                                            [np.argmax(Q[next_state_adj[0]][next_state_adj[1]])] - Q[state_adj[0]][state_adj[1]][action])

            if done:
                break

            action = next_action
            state_adj = next_state_adj

    return Q, stats
