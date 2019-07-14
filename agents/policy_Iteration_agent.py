import numpy as np

from utils.OneStepLookahead import one_step_lookahead


def policy_evaluation(policy, environment, discount_factor=1.0, theta=1e-9, max_iter=1e9):
    """
    Evaluate a policy given a deterministic environment.
    :param policy: Matrix of a size nSxnA, each cell represents a probability of taking action a in state s.
    :param environment: Initialized OpenAI gym environment object.
    :param discount_factor: MDP discount factor. Float in range from 0 to 1.
    :param theta: A threshold of a value function change.
    :param max_iter: Maximum number of iteration to prevent infinite loops.
    :return: A vector of size nS, which represent a value function for each state.
    """

    # Number of evaluation iterations
    evaluation_iterations = 1

    # Initialize a value function for each state as zero
    V = np.zeros(environment.nS)

    # Repeat until value change is below the threshold
    for i in range(int(max_iter)):

        # Initialize a change of value function as zero
        delta = 0

        # Iterate though each state
        for state in range(environment.nS):

            # Initial a new value of current state
            v = 0

            # Try all possible actions which can be taken from this state
            for action, action_probability in enumerate(policy[state]):

                # Evaluate how good each next state will be
                for state_probability, next_state, reward, terminated in environment.P[state][action]:

                    # Calculate the expected value
                    v += action_probability * state_probability * (reward + discount_factor * V[next_state])

            # Calculate the absolute change of value function
            delta = max(delta, np.abs(V[state] - v))

            # Update value function
            V[state] = v

        evaluation_iterations += 1

        # Terminate if value change is insignificant
        if delta < theta:
            print(f'Policy evaluated in {evaluation_iterations} iterations.')
            return V


def policy_iteration(environment, discount_factor=1.0, max_iter=1e9):
    """
    Policy iteration algorithm to solve MDP.
    :param environment: Initialized OpenAI gym environment object.
    :param discount_factor: MPD discount factor. Float in range from 0 to 1.
    :param max_iter: Maximum number of iterations to prevent infinite loops.
    :return: tuple(policy, V), which consist of an optimal policy matrix and value function for each state.
    """
    # Start with a random policy
    # num states x num actions / num actions
    policy = np.ones([environment.nS, environment.nA]) / environment.nA

    # Initialize counter of evaluated policies
    evaluated_policies = 1

    # Repeat until convergence or critical number of iterations reached
    for i in range(int(max_iter)):

        stable_policy = True

        # Evaluate current policy
        V = policy_evaluation(policy, environment, discount_factor=discount_factor)

        # Go through each state and try to improve actions that were taken
        for state in range(environment.nS):

            # Choose the best action in a current state under current policy
            current_action = np.argmax(policy[state])

            # Look one step ahead and evaluate if current action is optimal
            # We will try every possible action in a current state
            action_value = one_step_lookahead(environment, state, V, discount_factor)

            # Select a better action
            best_action = np.argmax(action_value)

            # If action didn't change
            if current_action != best_action:
                stable_policy = False

            # Greedy policy update
            policy[state] = np.eye(environment.nA)[best_action]

        evaluated_policies += 1

        # If the algorithm converged and policy is not changing anymore, than return final policy and value function
        if stable_policy:
            print(f'Evaluated {evaluated_policies} policies.')
            return policy, V
