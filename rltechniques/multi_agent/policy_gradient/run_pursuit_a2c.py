import numpy as np
from madrl_environments import ObservationBuffer
from madrl_environments.pursuit import PursuitEvade
from madrl_environments.pursuit.utils import TwoDMaps

from rltechniques.multi_agent import ENVParser
from rltechniques.multi_agent.ma_runner import Runner

ENV_OPTIONS = [
    ('n_evaders', int, 30, ''),
    ('n_pursuers', int, 8, ''),
    ('obs_range', int, 10, ''),
    ('map_size', str, '10,10', ''),
    ('map_type', str, 'rectangle', ''),
    ('n_catch', int, 2, ''),
    ('urgency', float, 0.0, ''),
    ('surround', int, 0, ''),
    ('map_file', str, None, ''),
    ('sample_maps', int, 0, ''),
    ('flatten', int, 0, ''),
    ('reward_mech', str, 'local', ''),
    ('catchr', float, 0.01, ''),
    ('term_pursuit', float, 5.0, ''),
    ('buffer_size', int, 1, ''),
    ('noid', str, None, ''),
]


def main(parser):
    args = parser.args

    if args.map_file:
        map_pool = np.load(args.map_file)
    else:
        if args.map_type == 'rectangle':
            env_map = TwoDMaps.rectangle_map(*map(int, args.map_size.split(',')))
        elif args.map_type == 'complex':
            env_map = TwoDMaps.complex_map(*map(int, args.map_size.split(',')))
        else:
            raise NotImplementedError()
        map_pool = [env_map]

    env = PursuitEvade(map_pool, n_evaders=args.n_evaders, n_pursuers=args.n_pursuers,
                       obs_range=args.obs_range, n_catch=args.n_catch, urgency_reward=args.urgency,
                       surround=bool(args.surround), sample_maps=bool(args.sample_maps),
                       flatten=bool(args.flatten), reward_mech=args.reward_mech, catchr=args.catchr,
                       term_pursuit=args.term_pursuit, include_id=not bool(args.noid))

    if args.buffer_size > 1:
        env = ObservationBuffer(env, args.buffer_size)

    runner = Runner(env, args)
    runner.start_training()


if __name__ == '__main__':
    main(ENVParser(ENV_OPTIONS))
