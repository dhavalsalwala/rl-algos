import numpy as np


def one_step_lookahead(environment, state, V, discount_factor):
    """
    Helper function to calculate a state-value function.
    :param environment: Initialized OpenAI gym environment object.
    :param state: Agent's state to consider (integer).
    :param V: The value to use as an estimator. Vector of length nS.
    :param discount_factor: MDP discount factor.
    :return: A vector of length nA containing the expected value of each action.
    """

    action_values = np.zeros(environment.nA)

    for action in range(environment.nA):

        for probability, next_state, reward, terminated in environment.P[state][action]:
            action_values[action] += probability * (reward + discount_factor * V[next_state])

    return action_values
