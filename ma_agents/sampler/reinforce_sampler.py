import itertools

import numpy as np
import rllab.misc.logger as logger
from rllab.algos import util
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.samplers.batch_ma_sampler import BatchMASampler


class ReinforceMASampler(BatchMASampler):

    def get_total_discounted_returns(self, rewards):
        discounted_episode_rewards = np.zeros_like(rewards)
        cumulative = 0.0
        for i in reversed(range(len(rewards))):
            cumulative = cumulative * self.algo.discount + rewards[i]
            discounted_episode_rewards[i] = cumulative

        return discounted_episode_rewards

    @overrides
    def process_samples(self, itr, paths):

        baselines = []
        returns = []
        paths = list(itertools.chain.from_iterable(paths))

        all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

        '''
        for index, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[index], 0)
            deltas = path["rewards"] + self.algo.discount * path_baselines[1:] - path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(deltas, self.algo.discount * self.algo.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])
        '''

        for index, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[index], 0)
            total_return = self.get_total_discounted_returns(path["rewards"])
            path["advantages"] = total_return - all_path_baselines[index]
            path["returns"] = total_return
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        ev = special.explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(returns)
        )

        observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
        actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
        rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
        returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
        advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
        env_info = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_info = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

        if self.algo.center_adv:
            advantages = util.center_advantages(advantages)

        if self.algo.positive_adv:
            advantages = util.shift_advantages_to_positive(advantages)

        average_discounted_return = np.mean([path["returns"][0] for path in paths])
        un_discounted_returns = [sum(path["rewards"]) for path in paths]
        ent = np.mean(self.algo.policy.distribution.entropy(agent_info))

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            returns=returns,
            advantages=advantages,
            env_info=env_info,
            agent_info=agent_info,
            paths=paths,
        )

        logger.log("fitting baseline...")
        self.algo.baseline.fit(paths)
        logger.log("fitted")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('NofTrajectories', samples_data['observations'].shape[0])
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        self.algo.avg_rewards = np.mean(un_discounted_returns)
        logger.record_tabular('AverageReturn', self.algo.avg_rewards)
        self.algo.total_rewards = np.sum(un_discounted_returns)
        logger.record_tabular('TotalReturn', self.algo.total_rewards)
        logger.record_tabular('MaxReturn', np.max(un_discounted_returns))
        logger.record_tabular('MinReturn', np.min(un_discounted_returns))
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(un_discounted_returns))

        return samples_data
