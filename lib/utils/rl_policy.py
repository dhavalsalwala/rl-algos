import numpy as np


def make_deep_epsilon_greedy_policy(env):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        env: open AI Gym environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def get_policy(state, approximater, epsilon):
        if np.random.uniform(0, 1) < (1 - epsilon):
            # take action according to the dq network
            action = np.argmax(approximater.predict(state))
        else:
            # take random action: EXPLORE
            action = env.action_space.sample()
        return action

    return get_policy


def make_random_policy(env):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        env: open AI Gym environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def get_policy():
        # take random action: EXPLORE
        action = env.action_space.sample()
        return action

    return get_policy


def make_stochastic_policy(nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        nA: open AI Gym environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def get_policy(state, nn):
        if state.ndim == 1:
            state = state.reshape(1, len(state))

        action_prob = nn.predict(state).flatten()
        return np.random.choice(nA, 1, p=action_prob)[0]

    return get_policy
