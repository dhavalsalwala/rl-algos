import itertools
import random
from collections import deque
import numpy as np
from rllab.misc import tensor_utils
from rllab.misc.overrides import overrides
from rllab.sampler import parallel_sampler
from sandbox.rocky.tf.samplers.batch_ma_sampler import BatchMASampler

from ma_agents.sampler import base_sampler


class ExpReplayMASampler(BatchMASampler):

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

    def obtain_random_samples(self, pre_trained_size):
        base_sampler.sample_random_paths(
            max_samples=pre_trained_size,
            sampler=self,
            max_path_length=self.algo.max_path_length,
            ma_mode=self.algo.ma_mode,
            scope=self.algo.scope, )

    @overrides
    def obtain_samples(self, itr):
        cur_policy_params = self.algo.policy.get_param_values()
        if hasattr(self.algo.env, "get_param_values"):
            cur_env_params = self.algo.env.get_param_values()
        else:
            cur_env_params = None

        paths = base_sampler.sample_paths(
            policy_params=cur_policy_params,
            env_params=cur_env_params,
            max_samples=self.algo.batch_size,
            sampler=self,
            max_path_length=self.algo.max_path_length,
            ma_mode=self.algo.ma_mode,
            scope=self.algo.scope, )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated

    @overrides
    def process_samples(self, itr, paths):

        paths = list(itertools.chain.from_iterable(paths))
        if not self.algo.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            next_observations = tensor_utils.concat_tensor_list([path["next_observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            done = tensor_utils.concat_tensor_list([path["done"] for path in paths])

            target = []

            next_q_value = self.algo.policy.get_actions(next_observations)[1]['prob']
            next_q_value_target = self.algo.target_policy.get_actions(next_observations)[1]['prob']
            for i in range(self.algo.batch_size):
                action = np.argmax(next_q_value[i])
                if done[i]:
                    target.append(rewards[i])
                else:
                    target.append(rewards[i] + self.algo.discount * next_q_value_target[i][action])

            samples_data = dict(
                observations=observations,
                next_observations=next_observations,
                actions=actions,
                rewards=rewards,
                target=np.array([each for each in target]),
                done=done,
                paths=paths,
            )

        else:
            pass

        return samples_data



