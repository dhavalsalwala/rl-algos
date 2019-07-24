<p align="center"><img width="90%" src="https://github.com/dhavalsalwala/rl-algos/tree/master/rltechniques/multi_agent/resources/pursuit_evasion.png"></p>


### Multi-Agent Deep Reinforcment Learning

This is the multi-agent section of the Reinforcement Learning repository available at https://github.com/dhavalsalwala/rl-algos

### Environment description
  
  - Pursuit Evasion
 
<p align="center"><img width="90%" src="https://github.com/dhavalsalwala/rl-algos/tree/master/rltechniques/multi_agent/resources/pursuit_evasion.png"></p>
  
### Setup
  - Add directories to PYTHONPATH
            
        export PYTHONPATH=$(pwd):$(pwd)/modules/MADRL:$(pwd)/modules/MADRL/rltools:$(pwd)/modules/MADRL/rllab:$PYTHONPATH
  
### RL Algos
  - DQN
  
        python3 rltechniques/multi_agent/dqn/run_pursuit.py --algo=dqn --control decentralized --policy_hidden 100,50,25 --n_iter 10 --batch_size 32 --replay_pre_trained_size 1000
  
  - REINFORCE
  
        python3 rltechniques/multi_agent/policy_gradient/run_pursuit.py --algo=vpg --control decentralized --policy_hidden 100,50,25 --n_iter 1000 --batch_size 1000

