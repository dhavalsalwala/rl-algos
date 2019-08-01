import numpy as np

from utils.one_step_lookahead import step_into


def value_iteration(environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
    """
    Value Iteration algorithm to solve MDP.
    :param environment: Initialized OpenAI environment object.
    :param theta: Stopping threshold. If the value of all states changes less than theta in one iteration - we are done.
    :param discount_factor: MDP discount factor.
    :param max_iterations: Maximum number of iterations that can be ever performed (to prevent infinite loops).
    :return: tuple (policy, V) which contains optimal policy and optimal value function.
    """

    # Initialize state-value function with zeros for each environment state
    V = np.zeros(environment.nS)

    for i in range(int(max_iterations)):

        # Early stopping condition
        delta = 0

        # Update each state
        for state in range(environment.nS):

            # Do a one-step lookahead to calculate state-action values
            action_value = step_into(environment, state, V, discount_factor)

            # Select best action to perform based on the highest state-action value
            best_action_value = np.max(action_value)

            # Calculate change in value
            delta = max(delta, np.abs(V[state] - best_action_value))

            # Update the value function for current state
            V[state] = best_action_value

        # Check if we can stop
        if delta < theta:
            print('Value-iteration converged at iteration#{i}.')
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([environment.nS, environment.nA])

    for state in range(environment.nS):

        # One step lookahead to find the best action for this state
        action_value = step_into(environment, state, V, discount_factor)

        # Select best action based on the highest state-action value
        best_action = np.argmax(action_value)

        # Update the policy to perform a better action at a current state
        policy[state, best_action] = 1.0

    return policy, V
