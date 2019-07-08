import sys
import time

import gym
import numpy as np

from agents.policy_Iteration_agent import policy_iteration

# Action mappings
action_mapping = {
    0: '\u2190',  # LEFT
    1: '\u2193',  # DOWN
    2: '\u2192',  # RIGHT
    3: '\u2191',  # UP
}


def play_episodes(environment, n_episodes, policy):
    wins = 0
    total_reward = 0

    for episode in range(n_episodes):
        environment.render()
        terminated = False
        state = environment.reset()
        print("*** Episode: ", episode)
        st=[]
        st.append(state)
        while not terminated:
            environment.render()

            # Select best action to perform in a current state
            action = np.argmax(policy[state])

            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, info = environment.step(action)

            st.append(next_state)
            # Summarize total reward
            total_reward += reward

            # Update current state
            state = next_state

            # Calculate number of wins over episodes
            if terminated and reward == 1.0:
                wins += 1
                break

            time.sleep(1)
            sys.stdout.flush()

    average_reward = total_reward / n_episodes

    return wins, total_reward, average_reward


# Number of episodes to play
n_episodes = 10000

# Functions to find best policy
solvers = [
    ('Policy Iteration', policy_iteration)
]

for iteration_name, iteration_func in solvers:

    # Load a Frozen Lake environment
    environment = gym.make('FrozenLake-v0')

    # Search for an optimal policy using policy iteration
    policy, V = iteration_func(environment.env)

    print(f'\n Final policy derived using {iteration_name}:')
    c = ""
    for i in range(16):
        if i >= 10:
            c = c + str(i) + "   "
        else:
            c = c+str(i)+"    "
    print(c)
    print('    '.join([action_mapping[action] for action in np.argmax(policy, axis=1)]))

    # Apply best policy to the real environment
    wins, total_reward, average_reward = play_episodes(environment, n_episodes, policy)

    print(f'{iteration_name} :: number of wins over {n_episodes} episodes = {wins}')
    print(f'{iteration_name} :: average reward over {n_episodes} episodes = {average_reward} \n\n')
