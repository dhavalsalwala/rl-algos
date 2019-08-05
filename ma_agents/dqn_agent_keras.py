from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
import numpy as np
import random
from ma_agents.handler.dqn_keras_ma_handler import BatchMADQNKERAS


class MADQNKeras(BatchMADQNKERAS, Serializable):
    """
    Multi-agent Deep Queue Network.
    """

    def __init__(self, env, optimizer=None, optimizer_args=None, **kwargs):
        Serializable.quick_init(self, locals())
        super(MADQNKeras, self).__init__(env=env, **kwargs)

    @overrides
    def init_opt(self):

        self.env.seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    @overrides
    def optimize_policy(self, itr, samples_data):
        pass

    @overrides
    def get_itr_snapshot(self, itr):

        agents = []
        for agent_index in range(self.env.n):
            if self.env.agents[agent_index].adversary:
                agents.append({'adversary_agent': self.networks[agent_index].get_weights()})
            else:
                agents.append({'good_agent': self.networks[agent_index].get_weights()})

        return dict(
            itr=itr,
            agents=agents,
            rewards=self.stats.episode_rewards,
            losses=self.stats.episode_losses,
            epsilon=self.stats.episode_epsilon,
            time_steps=self.stats.episode_lengths)
