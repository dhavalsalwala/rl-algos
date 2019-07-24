### Multi-Agent Deep Reinforcment Learning

This is the multi-agent section of the Reinforcement Learning repository available at https://github.com/dhavalsalwala/rl-algos

### Environment description
  
  - Pursuit Evasion
  
### Setup
  - Add directories to PYTHONPATH
            
        export PYTHONPATH=$(pwd):$(pwd)/modules/MADRL:$(pwd)/modules/MADRL/rltools:$(pwd)/modules/MADRL/rllab:$PYTHONPATH
  
## RL Algos
  - DQN
  
        python3 rltechniques/multi_agent/dqn/run_pursuit.py --algo=dqn --control decentralized --policy_hidden 100,50,25 --n_iter 10 --batch_size 32 --replay_pre_trained_size 1000

