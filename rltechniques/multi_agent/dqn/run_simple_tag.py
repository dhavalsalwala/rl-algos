from make_env import make_env
from rltechniques.multi_agent import ENVParser
from rltechniques.multi_agent.ma_runner import Runner

ENV_OPTIONS = [
    ('env', str, 'simple_tag', ''),
    ('benchmark', bool, True, ''),
]


def main(parser):
    args = parser.args
    env = make_env(args.env, args.benchmark)
    runner = Runner(env, args)
    runner.start_training()


if __name__ == '__main__':
    main(ENVParser(ENV_OPTIONS))
