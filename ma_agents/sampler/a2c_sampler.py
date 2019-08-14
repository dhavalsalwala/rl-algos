import itertools

import numpy as np
from rllab.misc import logger
from rllab.misc import tensor_utils
from rllab.misc.overrides import overrides
from rllab.sampler import parallel_sampler, ma_sampler
from sandbox.rocky.tf.samplers.batch_ma_sampler import BatchMASampler


class A2CMASampler(BatchMASampler):

    @overrides
    def obtain_samples(self, itr):
        if self.algo.ma_mode == 'concurrent':
            cur_policy_params = [policy.get_param_values() for policy in self.algo.policies]
        else:
            cur_policy_params = self.algo.policy.get_param_values()
        if hasattr(self.algo.env, "get_param_values"):
            cur_env_params = self.algo.env.get_param_values()
        else:
            cur_env_params = None
        paths = ma_sampler.sample_paths_a2c(
            policy_params=cur_policy_params,
            env_params=cur_env_params,
            max_samples=self.algo.max_path_length*len(self.algo.env.agents),
            max_path_length=self.algo.max_path_length,
            ma_mode=self.algo.ma_mode,
            scope=self.algo.scope,)
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated

    def get_target_returns(self, rewards, done, value_next):
        returns = np.append(np.zeros_like(rewards), np.array([value_next]), axis=-1)
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.algo.discount * returns[t + 1] * (1. - done[t])
        return returns[:-1]

    @overrides
    def process_samples(self, itr, paths):

        paths = list(itertools.chain.from_iterable(paths))

        td_target = []
        td_error_advantage = []

        for agent in paths:

            R = 0
            if agent['done_list'][-1]:
                R = self.algo.value_estimator.predict([agent['next_observations'][-1]])

            # value_next = self.algo.value_estimator.predict([agent['next_observations'][-1]])
            values = self.algo.value_estimator.predict(agent['observations'])

            n = len(agent['observations'])
            td = np.zeros(n)
            adv = np.zeros(n)
            for i in range(n - 1, -1, -1):
                R = agent['rewards'][i] + self.algo.discount * R
                td[i] = R
                adv[i] = R - values[i]

            # td_target_tmp = self.get_target_returns(agent['rewards'], agent['done_list'], value_next)
            # td_error_advantage_tmp = td_target_tmp - values

            td_target.append(td)
            td_error_advantage.append(adv)

        samples_data = dict(
            observations=tensor_utils.concat_tensor_list([path["observations"] for path in paths]),
            actions=tensor_utils.concat_tensor_list([path["actions"] for path in paths]),
            rewards=tensor_utils.concat_tensor_list([path["rewards"] for path in paths]),
            env_infos=tensor_utils.concat_tensor_dict_list([path["env_info"] for path in paths]),
            agent_info=tensor_utils.concat_tensor_dict_list([path["agent_info"] for path in paths]),
            td_target=tensor_utils.concat_tensor_list(td_target),
            td_error=tensor_utils.concat_tensor_list(td_error_advantage),
            paths=paths,
        )

        un_discounted_returns = [sum(path["rewards"]) for path in paths]

        ent = np.mean(self.algo.policy.distribution.entropy(samples_data['agent_info']))
        logger.log("Optimising critic function...")
        self.algo.value_loss = self.algo.value_estimator.update(samples_data['observations'], samples_data['td_target'])

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('NofTrajectories', samples_data['observations'].shape[0])
        logger.record_tabular('AverageReturn', np.mean(un_discounted_returns))
        logger.record_tabular('TotalReturn', np.sum(un_discounted_returns))
        logger.record_tabular('MaxReturn', np.max(un_discounted_returns))
        logger.record_tabular('MinReturn', np.min(un_discounted_returns))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(un_discounted_returns))
        logger.record_tabular("ValueLoss", self.algo.value_loss)

        return samples_data



