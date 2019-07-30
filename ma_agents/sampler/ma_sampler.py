import time

import numpy as np
from rllab.misc import tensor_utils
from rllab.sampler.ma_sampler import _worker_set_policy_params, _worker_set_env_params
from rllab.sampler.parallel_sampler import (_get_scoped_G)
from rllab.sampler.stateful_pool import singleton_pool


def dec_roll_out_random(env, agents, sampler, max_path_length=1000, animated=False, speedup=1):
    """Decentralized roll out"""
    n_agents = len(env.agents)

    observation_list = env.reset()
    assert len(observation_list) == n_agents, "{} != {}".format(len(observation_list), n_agents)
    agents.reset(dones=[True for _ in range(n_agents)])

    path_length = 0
    while path_length < max_path_length:

        observations = [[] for _ in range(n_agents)]
        next_observations = [[] for _ in range(n_agents)]
        actions = [[] for _ in range(n_agents)]
        rewards = [[] for _ in range(n_agents)]
        done_list = [[] for _ in range(n_agents)]
        agent_info = [[] for _ in range(n_agents)]
        env_info_list = [[] for _ in range(n_agents)]

        action_list, agent_info_list = agents.get_actions(observation_list)
        agent_info_list = tensor_utils.split_tensor_dict_list(agent_info_list)

        # For each agent
        for index, observation in enumerate(observation_list):
            observations[index].append(observation)
            actions[index].append(env.action_space.flatten(action_list[index]))
            if agent_info_list is None:
                agent_info[index].append({})
            else:
                agent_info[index].append(agent_info_list[index])

        next_observation_list, reward_list, done, env_info = env.step(np.asarray(action_list))

        # For each agent
        for index, reward in enumerate(reward_list):
            next_observations[index].append(next_observation_list[index])
            rewards[index].append(reward)
            env_info_list[index].append(env_info)
            done_list[index].append(done)

        sampler.add_to_replay_memory((observations, actions, rewards, next_observations, done_list, agent_info, env_info_list))

        path_length += 1

        if done:
            break

    return path_length


def dec_roll_out(env, agents, sampler, max_path_length, animated=False, speedup=1):
    """Decentralized roll out"""
    n_agents = len(env.agents)

    if sampler.done:
        sampler.done = False
        observation_list = env.reset()
        assert len(observation_list) == n_agents, "{} != {}".format(len(observation_list), n_agents)
        agents.reset(dones=[True for _ in range(n_agents)])
    else:
        observation_list = sampler.old_observation_list

    if animated:
        env.render()

    observations = [[] for _ in range(n_agents)]
    next_observations = [[] for _ in range(n_agents)]
    actions = [[] for _ in range(n_agents)]
    rewards = [[] for _ in range(n_agents)]
    done_list = [[] for _ in range(n_agents)]
    agent_info = [[] for _ in range(n_agents)]
    env_info_list = [[] for _ in range(n_agents)]

    action_list, agent_info_list = agents.get_actions(observation_list)
    agent_info_list = tensor_utils.split_tensor_dict_list(agent_info_list)

    # For each agent
    for index, observation in enumerate(observation_list):
        observations[index].append(observation)
        actions[index].append(env.action_space.flatten(action_list[index]))
        if agent_info_list is None:
            agent_info[index].append({})
        else:
            agent_info[index].append(agent_info_list[index])

    next_observation_list, reward_list, done, env_info = env.step(np.asarray(action_list))

    # For each agent
    for index, reward in enumerate(reward_list):
        next_observations[index].append(next_observation_list[index])
        rewards[index].append(reward)
        sampler.algo.total_episodic_rewards[index].append(reward)
        env_info_list[index].append(env_info)
        done_list[index].append(done)

    sampler.add_to_replay_memory((observations, actions, rewards, next_observations, done_list, agent_info, env_info_list))

    samples = sampler.get_sample_from_replay_memory()
    observations = [[] for _ in range(n_agents)]
    next_observations = [[] for _ in range(n_agents)]
    actions = [[] for _ in range(n_agents)]
    rewards = [[] for _ in range(n_agents)]
    done_list = [[] for _ in range(n_agents)]
    agent_info = [[] for _ in range(n_agents)]
    env_info_list = [[] for _ in range(n_agents)]
    for sample in samples:
        for index in range(n_agents):
            observations[index].append(sample[0][index][0])
            actions[index].append(sample[1][index])
            rewards[index].append(sample[2][index])
            next_observations[index].append(sample[3][index][0])
            done_list[index].append(sample[4][index])
            agent_info[index].append(sample[5][index][0])
            env_info_list[index].append(sample[6][index][0])

    sampler.old_observation_list = next_observation_list
    sampler.done = done

    if animated:
        env.render()
        time_step = 0.05
        time.sleep(time_step / speedup)

    if animated:
        env.render()

    return [
        dict(
            observations=tensor_utils.stack_tensor_list(observations[i]),
            next_observations=tensor_utils.stack_tensor_list(next_observations[i]),
            actions=np.reshape(tensor_utils.stack_tensor_list(actions[i]), (-1, env.action_space.n)),
            rewards=tensor_utils.stack_tensor_list(rewards[i]),
            done=tensor_utils.stack_tensor_list(done_list[i]),
            agent_info=tensor_utils.stack_tensor_dict_list(agent_info[i]),
            env_info=tensor_utils.stack_tensor_dict_list(env_info_list[i]), ) for i in range(n_agents)
    ]


def dec_roll_out_once(env, agents, sampler, max_path_length, animated=False, speedup=1):
    """Decentralized roll out"""
    n_agents = len(env.agents)

    if sampler.done:
        sampler.done = False
        observation_list = env.reset()
        assert len(observation_list) == n_agents, "{} != {}".format(len(observation_list), n_agents)
        agents.reset(dones=[True for _ in range(n_agents)])
    else:
        observation_list = sampler.old_observation_list

    if animated:
        env.render()

    observations = [[] for _ in range(n_agents)]
    next_observations = [[] for _ in range(n_agents)]
    actions = [[] for _ in range(n_agents)]
    rewards = [[] for _ in range(n_agents)]
    done_list = [[] for _ in range(n_agents)]
    agent_info = [[] for _ in range(n_agents)]
    env_info_list = [[] for _ in range(n_agents)]

    action_list, agent_info_list = agents.get_actions(observation_list)
    agent_info_list = tensor_utils.split_tensor_dict_list(agent_info_list)

    # For each agent
    for index, observation in enumerate(observation_list):
        observations[index].append(observation)
        actions[index].append(env.action_space.flatten(action_list[index]))
        if agent_info_list is None:
            agent_info[index].append({})
        else:
            agent_info[index].append(agent_info_list[index])

    next_observation_list, reward_list, done, env_info = env.step(np.asarray(action_list))

    # For each agent
    for index, reward in enumerate(reward_list):
        next_observations[index].append(next_observation_list[index])
        rewards[index].append(reward)
        sampler.algo.total_episodic_rewards[index].append(reward)
        env_info_list[index].append(env_info)
        done_list[index].append(done)

    sampler.old_observation_list = next_observation_list
    sampler.done = done

    if animated:
        env.render()
        time_step = 0.05
        time.sleep(time_step / speedup)

    if animated:
        env.render()

    return [
        dict(
            observations=tensor_utils.stack_tensor_list(observations[i]),
            next_observations=tensor_utils.stack_tensor_list(next_observations[i]),
            actions=np.reshape(tensor_utils.stack_tensor_list(actions[i]), (-1, env.action_space.n)),
            rewards=tensor_utils.stack_tensor_list(rewards[i]),
            done=tensor_utils.stack_tensor_list(done_list[i]),
            agent_info=tensor_utils.stack_tensor_dict_list(agent_info[i]),
            env_info=tensor_utils.stack_tensor_dict_list(env_info_list[i]), ) for i in range(n_agents)
    ]


def _worker_collect_path_random_one_env(G, max_path_length, ma_mode, sampler, scope=None):
    G = _get_scoped_G(G, scope)
    if ma_mode == 'decentralized':
        paths = dec_roll_out_random(G.env, G.policy, sampler, max_path_length)
        return paths, paths
    else:
        raise NotImplementedError("incorrect rollout type")


def _worker_collect_path_one_env(G, max_path_length, ma_mode, sampler, scope=None):
    G = _get_scoped_G(G, scope)
    if ma_mode == 'decentralized':
        paths = dec_roll_out(G.env, G.policy, sampler, max_path_length)
        lengths = [len(path['rewards']) for path in paths]
        return paths, sum(lengths)
    else:
        raise NotImplementedError("incorrect rollout type")


def _worker_collect_path_one_env_once(G, max_path_length, ma_mode, sampler, scope=None):
    G = _get_scoped_G(G, scope)
    if ma_mode == 'decentralized':
        paths = dec_roll_out_once(G.env, G.policy, sampler, max_path_length)
        lengths = [len(path['rewards']) for path in paths]
        return paths, sum(lengths)
    else:
        raise NotImplementedError("incorrect rollout type")


def sample_paths(policy_params, max_samples, ma_mode, sampler, max_path_length=np.inf, env_params=None,
                 scope=None):
    singleton_pool.run_each(_worker_set_policy_params,
                            [(policy_params, ma_mode, scope)] * singleton_pool.n_parallel)
    if env_params is not None:
        singleton_pool.run_each(_worker_set_env_params,
                                [(env_params, scope)] * singleton_pool.n_parallel)

    return singleton_pool.run_collect(_worker_collect_path_one_env, threshold=max_samples,
                                      args=(max_path_length, ma_mode, sampler, scope), show_prog_bar=False)


def sample_paths_once(policy_params, max_samples, ma_mode, sampler, max_path_length=np.inf, env_params=None,
                 scope=None):
    singleton_pool.run_each(_worker_set_policy_params,
                            [(policy_params, ma_mode, scope)] * singleton_pool.n_parallel)
    if env_params is not None:
        singleton_pool.run_each(_worker_set_env_params,
                                [(env_params, scope)] * singleton_pool.n_parallel)

    return singleton_pool.run_collect(_worker_collect_path_one_env_once, threshold=max_samples,
                                      args=(max_path_length, ma_mode, sampler, scope), show_prog_bar=False)


def sample_random_paths(max_samples, ma_mode, sampler, max_path_length=np.inf,
                        scope=None):
    singleton_pool.run_collect(_worker_collect_path_random_one_env, threshold=max_samples,
                                      args=(max_path_length, ma_mode, sampler, scope), show_prog_bar=True)
