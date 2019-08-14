import argparse
import ast
import datetime
import sys
import uuid

import dateutil
import dateutil.tz

from rltechniques.multi_agent import *


class ARGParser(object):
    DEFAULT_OPTS = [
        ('discount', float, 0.99, ''),
        ('gae_lambda', float, 0.99, ''),
        ('n_iter', int, 25000, ''),
    ]

    DEFAULT_POLICY_OPTS = [
        ('control', str, 'decentralized', ''),
        ('recurrent', str, None, ''),
        ('baseline_type', str, 'linear', ''),
    ]

    def __init__(self, env_options, **kwargs):
        self._env_options = env_options
        parser = argparse.ArgumentParser(description='DQN Argument Parser')

        getattr(self, "set_param_values")(self._env_options, **kwargs)

    @staticmethod
    def update_argument_parser(parser, options, **kwargs):
        kwargs = kwargs.copy()
        for (name, typ, default, desc) in options:
            flag = "--" + name
            if flag in parser._option_string_actions.keys():
                print("warning: already have option %s. skipping" % name)
            else:
                parser.add_argument(flag, type=typ, default=kwargs.pop(name, default), help=desc or
                                                                                            " ")
        if kwargs:
            raise ValueError("options %s ignored" % kwargs)

    def set_param_values(self, env_options, **kwargs):
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        rand_id = str(uuid.uuid4())[:5]
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
        default_exp_name = 'experiment_%s_%s' % (timestamp, rand_id)

        parser = argparse.ArgumentParser()
        parser.add_argument('--exp_name', type=str, default=default_exp_name)
        self.update_argument_parser(parser, self.DEFAULT_OPTS)
        self.update_argument_parser(parser, self.DEFAULT_POLICY_OPTS)

        parser.add_argument(
            '--algo', type=str, default='',
            help='Add tf or th to the algo name to run tensorflow or theano version')

        parser.add_argument('--max_path_length', type=int, default=256)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--n_parallel', type=int, default=1)
        parser.add_argument('--resume_from', type=str, default=None,
                            help='Name of the pickle file to resume experiment from.')

        parser.add_argument('--epoch_length', type=int, default=1000)
        parser.add_argument('--min_pool_size', type=int, default=10000)
        parser.add_argument('--replay_pool_size', type=int, default=500000)
        parser.add_argument('--replay_pre_trained_size', type=int, default=10)
        parser.add_argument('--eval_samples', type=int, default=50000)
        parser.add_argument('--qfunc_lr', type=float, default=1e-3)
        parser.add_argument('--policy_lr', type=float, default=1e-4)
        parser.add_argument('--target_network_update', type=int, default=1000)
        parser.add_argument('--save_param_update', type=int, default=125)

        parser.add_argument('--feature_net', type=str, default=None)
        parser.add_argument('--feature_output', type=int, default=16)
        parser.add_argument('--feature_hidden', type=comma_sep_ints, default='400,300')
        parser.add_argument('--policy_hidden', type=comma_sep_ints, default='512')

        parser.add_argument('--conv', type=bool, default=True)
        parser.add_argument('--conv_filters', type=comma_sep_ints, default='3,3,3')
        parser.add_argument('--conv_channels', type=comma_sep_ints, default='32,64,64')
        parser.add_argument('--conv_strides', type=comma_sep_ints, default='1,1,1')
        parser.add_argument('--conv_pads', type=str, default='SAME')
        parser.add_argument('--batch_normalization', type=bool, default=True)

        parser.add_argument('--min_std', type=float, default=1e-6)
        parser.add_argument('--exp_strategy', type=str, default='ou')
        parser.add_argument('--exp_noise', type=float, default=0.3)

        parser.add_argument('--step_size', type=float, default=0.01, help='max kl wall limit')

        parser.add_argument('--log_dir', type=str, required=False)
        parser.add_argument('--tabular_log_file', type=str, default='progress.csv',
                            help='Name of the tabular log file (in csv).')
        parser.add_argument('--text_log_file', type=str, default='debug.log',
                            help='Name of the text log file (in pure text).')
        parser.add_argument('--params_log_file', type=str, default='params.json',
                            help='Name of the parameter log file (in json).')
        parser.add_argument('--seed', type=int, help='Random seed for numpy')
        parser.add_argument('--args_data', type=str, help='Pickled data for stub objects')
        parser.add_argument('--snapshot_mode', type=str, default='all',
                            help='Mode to save the snapshot. Can be either "all" '
                                 '(all iterations will be saved), "last" (only '
                                 'the last iteration will be saved), or "none" '
                                 '(do not save snapshots)')
        parser.add_argument(
            '--log_tabular_only', type=ast.literal_eval, default=False,
            help='Whether to only print the tabular log information (in a horizontal format)')

        parser.add_argument('--q_learning_flag', type=bool, default=True)

        self.update_argument_parser(parser, env_options, **kwargs)
        self.args = parser.parse_known_args(
            [arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])[0]