# continuous-safety-gym

A suite with gym environments for safe RL applications with continuous cost functions.

Environments were adapted from:

+ https://github.com/AgrawalAmey/safe-explorer
+ https://github.com/SvenGronauer/Bullet-Safety-Gym
+ https://github.com/zisikons/multiagent-particle-envs


## Installation

```
git clone https://github.com/FelippeRoza/continuous-safety-gym.git

cd continuous-safety-gym

pip install -e .
```

## Getting started

```
import gymnasium as gym
import bullet_safety_gym
env = gym.make('SafetyCarGather-v0')
```


