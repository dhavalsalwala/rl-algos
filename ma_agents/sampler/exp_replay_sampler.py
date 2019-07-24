import itertools
import pickle
import random
import time
from collections import deque

import numpy as np
import tensorflow as tf
from rllab.misc import logger, tensor_utils
from rllab.sampler import parallel_sampler
from rllab.sampler.base import BaseSampler
from rllab.sampler.parallel_sampler import (_get_scoped_G)
from rllab.sampler.stateful_pool import singleton_pool


def worker_init_tf(G):
    G.sess = tf.Session()
    G.sess.__enter__()


def worker_init_tf_vars(G):
    G.sess.run(tf.initialize_all_variables())


class ExpReplayMASampler(BaseSampler):

    def __init__(self, algo=None, **kwargs):

        self.replay_memory = deque(maxlen=100000)
        self.old_observation_list = None
        self.done = True
        super(ExpReplayMASampler, self).__init__(algo, **kwargs)

    def add_to_replay_memory(self, sample):
        self.replay_memory.append(sample)

    def replay_memory_size(self):
        return len(self.replay_memory)

    def get_sample_from_replay_memory(self):
        n = min(round(self.algo.batch_size/len(self.algo.env.agents)), len(self.replay_memory))
        return random.sample(self.replay_memory, n)

    def start_worker(self):
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf)
        self.populate_task(self.algo.env, self.algo.q_network, self.algo.ma_mode)
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf_vars)

    def shutdown_worker(self):
        self.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr):
        cur_policy_params = self.algo.q_network.get_param_values()
        if hasattr(self.algo.env, "get_param_values"):
            cur_env_params = self.algo.env.get_param_values()
        else:
            cur_env_params = None

        paths = self.sample_paths(
            policy_params=cur_policy_params,
            env_params=cur_env_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            ma_mode=self.algo.ma_mode,
            scope=self.algo.scope, )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated

    def process_samples(self, itr, paths):

        paths = list(itertools.chain.from_iterable(paths))
        if not self.algo.q_network.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            next_observations = tensor_utils.concat_tensor_list([path["next_observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            done = tensor_utils.concat_tensor_list([path["done"] for path in paths])
            env_info = tensor_utils.concat_tensor_dict_list([path["env_info"] for path in paths])
            agent_info = tensor_utils.concat_tensor_dict_list([path["agent_info"] for path in paths])

            samples_data = dict(
                observations=observations,
                next_observations=next_observations,
                actions=actions,
                rewards=rewards,
                done=done,
                env_infos=env_info,
                agent_info=agent_info,
                paths=paths,
            )
        else:
            pass

        return samples_data

    def populate_random(self, env, agents, max_path_length=1000, animated=False, speedup=1):
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

            self.add_to_replay_memory((observations, actions, rewards, next_observations, done_list, agent_info, env_info_list))

            if done:
                observation_list = env.reset()
            else:
                observation_list = next_observation_list

            path_length += 1

        return self.replay_memory

    def dec_roll_out(self, env, agents, animated=False, speedup=1):
        """Decentralized roll out"""
        n_agents = len(env.agents)

        if self.done:
            self.done = False
            observation_list = env.reset()
            assert len(observation_list) == n_agents, "{} != {}".format(len(observation_list), n_agents)
            agents.reset(dones=[True for _ in range(n_agents)])
        else:
            observation_list = self.old_observation_list

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
            self.algo.total_episodic_rewards[index].append(reward)
            env_info_list[index].append(env_info)
            done_list[index].append(done)

        self.add_to_replay_memory((observations, actions, rewards, next_observations, done_list, agent_info, env_info_list))

        samples = self.get_sample_from_replay_memory()
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

        self.old_observation_list = next_observation_list
        self.done = done

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
                actions=tensor_utils.stack_tensor_list(actions[i]),
                rewards=tensor_utils.stack_tensor_list(rewards[i]),
                done=tensor_utils.stack_tensor_list(done_list[i]),
                agent_info=tensor_utils.stack_tensor_dict_list(agent_info[i]),
                env_info=tensor_utils.stack_tensor_dict_list(env_info_list[i]), ) for i in range(n_agents)
        ]

    def _worker_populate_task(self, G, env, policy, ma_mode, scope=None):
        # TODO: better term for both policy/policies
        G = _get_scoped_G(G, scope)
        G.env = pickle.loads(env)
        if ma_mode == 'concurrent':
            G.policies = pickle.loads(policy)
            assert isinstance(G.policies, list)
        else:
            G.policy = pickle.loads(policy)

    def _worker_terminate_task(self, G, scope=None):
        G = _get_scoped_G(G, scope)
        if getattr(G, "env", None):
            G.env.terminate()
            G.env = None
        if getattr(G, "policy", None):
            G.policy.terminate()
            G.policy = None
        if getattr(G, "policies", None):
            for policy in G.policies:
                policy.terminate()
            G.policies = None
        if getattr(G, "sess", None):
            G.sess.close()
            G.sess = None

    def populate_task(self, env, policy, ma_mode, scope=None):
        logger.log("Populating workers...")
        logger.log("ma_mode={}".format(ma_mode))
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(self._worker_populate_task,
                                    [(pickle.dumps(env), pickle.dumps(policy), ma_mode, scope)] *
                                    singleton_pool.n_parallel)
        else:
            # avoid unnecessary copying
            G = _get_scoped_G(singleton_pool.G, scope)
            G.env = env
            G.policy = policy
        logger.log("Populated")

    def terminate_task(self, scope=None):
        singleton_pool.run_each(self._worker_terminate_task, [(scope,)] * singleton_pool.n_parallel)

    def _worker_set_env_params(self, G, params, scope=None):
        G = _get_scoped_G(G, scope)
        G.env.set_param_values(params)

    def _worker_set_policy_params(self, G, params, ma_mode, scope=None):
        G = _get_scoped_G(G, scope)
        G.policy.set_param_values(params)

    def _worker_collect_path_one_env(self, G, max_path_length, ma_mode, scope=None):
        G = _get_scoped_G(G, scope)
        if ma_mode == 'decentralized':
            paths = self.dec_roll_out(G.env, G.policy)
            lengths = [len(path['rewards']) for path in paths]
            return paths, sum(lengths)
        else:
            raise NotImplementedError("incorrect roll out type")

    def _worker_collect_random_path_one_env(self, G, max_path_length, ma_mode, scope=None):
        G = _get_scoped_G(G, scope)
        if ma_mode == 'decentralized':
            paths = self.populate_random(G.env, G.policy, max_path_length)
            return paths, len(paths)
        else:
            raise NotImplementedError("incorrect roll out type")

    def sample_paths(self, policy_params, max_samples, ma_mode, max_path_length=np.inf,
                     env_params=None, scope=None):

        singleton_pool.run_each(self._worker_set_policy_params, [(policy_params, scope)] * singleton_pool.n_parallel)
        if env_params is not None:
            singleton_pool.run_each(self._worker_set_env_params, [(env_params, scope)] * singleton_pool.n_parallel)
        return singleton_pool.run_collect(self._worker_collect_path_one_env, threshold=max_samples,
                                          args=(max_path_length, ma_mode, scope), show_prog_bar=False)

    def sample_random_paths(self, max_samples, ma_mode, max_path_length=np.inf,
                            scope=None):

        return singleton_pool.run_collect(self._worker_collect_random_path_one_env, threshold=max_samples,
                                          args=(max_path_length, ma_mode, scope), show_prog_bar=False)

