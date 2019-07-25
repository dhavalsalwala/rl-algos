from __future__ import absolute_import, print_function

import argparse
import os
import os.path
import pprint

import numpy as np
from madrl_environments import ObservationBuffer
from madrl_environments.pursuit import PursuitEvade
from madrl_environments.pursuit.utils import TwoDMaps
from rllab.misc.tabulate import tabulate
from vis import Evaluator, FileHandler

from rltechniques.multi_agent import Visualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('--vid', type=str, default='/tmp/madrl.mp4')
    parser.add_argument('--deterministic', action='store_true', default=False)
    parser.add_argument('--heuristic', action='store_true', default=False)
    parser.add_argument('--evaluate', action='store_true', default=True)
    parser.add_argument('--n_trajs', type=int, default=20)
    parser.add_argument('--n_steps', type=int, default=20)
    parser.add_argument('--same_con_pol', action='store_true')
    args = parser.parse_args()

    fh = FileHandler(args.filename)

    if fh.train_args['map_file'] is not None:
        map_pool = np.load(
            os.path.join('', os.path.basename(fh.train_args[
                'map_file'])))
    else:
        if fh.train_args['map_type'] == 'rectangle':
            env_map = TwoDMaps.rectangle_map(*map(int, fh.train_args['map_size'].split(',')))
        elif args.map_type == 'complex':
            env_map = TwoDMaps.complex_map(*map(int, fh.train_args['map_size'].split(',')))
        else:
            raise NotImplementedError()
        map_pool = [env_map]

    env = PursuitEvade(map_pool, n_evaders=fh.train_args['n_evaders'],
                       n_pursuers=fh.train_args['n_pursuers'], obs_range=fh.train_args['obs_range'],
                       n_catch=fh.train_args['n_catch'], urgency_reward=fh.train_args['urgency'],
                       surround=bool(fh.train_args['surround']),
                       sample_maps=bool(fh.train_args['sample_maps']),
                       flatten=bool(fh.train_args['flatten']), reward_mech='global',
                       catchr=fh.train_args['catchr'], term_pursuit=fh.train_args['term_pursuit'])

    if fh.train_args['buffer_size'] > 1:
        env = ObservationBuffer(env, fh.train_args['buffer_size'])

    hpolicy = None
    if args.evaluate:
        minion = Evaluator(env, fh.train_args, args.n_steps, args.n_trajs, args.deterministic,
                           'heuristic' if args.heuristic else fh.mode)
        evr = minion(fh.filename, file_key=fh.file_key, same_con_pol=args.same_con_pol,
                     hpolicy=hpolicy)
        print(evr)
        print(tabulate(evr, headers='keys'))
    else:
        minion = Visualizer(env, fh.train_args, args.n_steps, args.n_trajs, args.deterministic,
                            fh.mode)
        rew, info = minion(fh.filename, file_key=fh.file_key, vid=args.vid)
        pprint.pprint(rew)
        pprint.pprint(info)


if __name__ == '__main__':
    main()
