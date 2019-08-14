from collections import namedtuple

import joblib
import numpy as np
import tensorflow as tf
from vis import PolicyLoad


def to_named_tuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = tonamedtuple(value)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def comma_sep_ints(s):
    if s:
        return list(map(int, s.split(",")))
    else:
        return []


class Visualizer(PolicyLoad):

    def __init__(self, *args, **kwargs):
        super(Visualizer, self).__init__(*args, **kwargs)

    def __call__(self, filename, **kwargs):
        vid = kwargs.pop('vid', None)
        if self.mode == 'rllab':
            tf.reset_default_graph()
            with tf.Session() as sess:

                data = joblib.load(filename)
                policy = data['q_network']
                if self.control == 'decentralized':
                    act_fns = lambda o: policy.get_actions(o)[0]
                    rew, trajinfo = self.env.wrapped_env.env.animate(act_fn=act_fns,
                                                                     nsteps=self.max_traj_len,
                                                                     file_name=vid)
                    info = {key: np.sum(value) for key, value in trajinfo.items()}
                return rew, info
