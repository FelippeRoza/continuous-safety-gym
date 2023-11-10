'''
source: https://github.com/AgrawalAmey/safe-explorer
'''

import gymnasium as gym
from gymnasium.spaces import Box, Dict, flatten, flatten_space
import numpy as np
from numpy import linalg as LA
import argparse
import copy
import os
import sys
import yaml
import inspect
from pathlib import Path
from typing import Optional


class Namespacify(object):
    def __init__(self, name, in_dict):
        self.name = name

        for key in in_dict.keys():
            if isinstance(in_dict[key], dict):
                in_dict[key] = Namespacify(key, in_dict[key])
    
        self.__dict__.update(in_dict)

    def pprint(self, indent=0):
        print(f"{' ' * indent}{self.name}:")
        
        indent += 4
        
        for k,v in self.__dict__.items():
            if k == "name":
                continue
            if type(v) == Namespacify:
                v.pprint(indent)
            else:
                print(f"{' ' * indent}{k}: {v}")


class Config:
    _config = None

    @classmethod
    def _get_argument_groups(cls, arg_config):
        """Reads config.yml file and creates groups of arguments by flattening hierarchy"""
        argument_groups = []
        for group in arg_config:
            if "properties" in group.keys():
                sub_groups = cls._get_argument_groups(group["properties"])
                for sub_group in sub_groups:
                    if not sub_group.get("_is_root", False):
                        sub_group["name"] = f"{group['name']}_{sub_group['name']}".strip()
                        sub_group["help"] = f"{group.get('help', '')} {sub_group.get('help', '')}".strip()

                    for argument in sub_group.get("arguments", []):
                        argument["name"] = f"{sub_group['name']}_{argument['name']}".strip()
                        del argument["_is_root"]

                if all([x.get("_is_root", False) for x in sub_groups]):
                    argument_groups.append({"name": group["name"],
                                            "help": group.get("help", ""),
                                            "arguments": sub_groups})
                else:
                    argument_groups += sub_groups
            else:
                group["_is_root"] = True
                argument_groups.append(copy.deepcopy(group))

        return argument_groups

    @classmethod
    def _create_parser(cls, name, _help, argument_groups):
        """Creates argument parser from argument groups"""
        parser = argparse.ArgumentParser(prog=name,
                                         description=_help,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        for argument_group in argument_groups:
            group = parser.add_argument_group(argument_group["name"], argument_group.get('help', ''))
            for argument in argument_group.get("arguments", []):
                default_value = argument.get("default", None)
                options = {
                    "help": argument.get('help', ' '),
                    "default": default_value
                }
                
                if type(default_value) == bool:
                    options["action"] = 'store_true'
                else:
                    options["type"] = type(default_value) if default_value else None
                    if type(default_value) == list:
                        options["nargs"] = '+'

                group.add_argument(f"--{argument['name']}", **options)
        
        return parser
    
    @classmethod
    def _split_namespace(cls, parent_name, arg_config, parsed_dict):
        """Reestablish the hierarchy from parsed arguments"""
        group_namespaces = {}
        for group in arg_config:
            name = group["name"]
            if "properties" in group.keys():
                matching_arguments = {k[(len(name) + 1):]: v for k, v \
                                    in parsed_dict.items() if k.startswith(name)}
                group_namespaces[name] = cls._split_namespace(name, group["properties"], matching_arguments)
            else:
                group_namespaces[name] = parsed_dict[name]

        return Namespacify(parent_name, group_namespaces)

    @classmethod
    def _load_config(cls, args=None):
        """Reads the default config and parses the commandline args to generate config"""
        config_file_path = f"{Path(__file__).parent}/config_defaults.yml"
        config = yaml.load(open(config_file_path), Loader=yaml.FullLoader)
        argument_groups = cls._get_argument_groups(config["arguments"])
        parser = cls._create_parser(config["name"], config.get("help", ''), argument_groups)
        parsed_config = cls._split_namespace(config["name"], config["arguments"], parser.parse_args(args).__dict__)

        cls._config = parsed_config

    @classmethod
    def get(cls):
        if not cls._config:
            cls._load_config()

        return cls._config


class Spaceship(gym.Env):
    def __init__(self):

        self._config = Config.get().env.spaceship

        self._width = self._config.length if self._config.is_arena else 1
        self._episode_length = self._config.arena_episode_length \
            if self._config.is_arena else self._config.corridor_episode_length
        # Set the properties for spaces
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float64)
        self.dict_observation_space = Dict({
            'agent_position': Box(low=0, high=1, shape=(2,), dtype=np.float64),
            'agent_velocity': Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
            'target_position': Box(low=0, high=1, shape=(2,), dtype=np.float64)
        })
        self.observation_space = flatten_space(self.dict_observation_space)
        self.constraint_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        # Sets all the episode specific variables         
        self.reset()
        
    def get_info(self):
        constraints = self.get_constraint_values()
        info = {'cost': constraints, 'cost_max': max(constraints)}
        return info

    def reset(self, *, seed: Optional[int] = None, options={}):
        super().reset(seed=seed)
        self._velocity = np.zeros(2, dtype=np.float64)
        self._agent_position = \
            (np.asarray([self._width , self._config.length / 3]) - 2 * self._config.margin) * np.random.random(2) \
                 + self._config.margin
        self._target_position = \
            (np.asarray([self._width , self._config.length]) - 2 * self._config.margin) * np.random.random(2) \
                 + self._config.margin
        self._current_time = 0.        

        return self.step(np.zeros(2))[0], self.get_info()

    def _get_reward(self):
        if self._config.enable_reward_shaping and self._is_agent_outside_shaping_boundary():
            reward = -1000
        elif LA.norm(self._agent_position - self._target_position) < self._config.target_radius:
            reward = 1000
        else:
            reward = 0

        return reward
    
    def _move_agent(self, acceleration):
        # Assume spaceship frequency to be one
        self._agent_position += self._velocity * self._config.frequency_ratio \
                                + 0.5 * acceleration * self._config.frequency_ratio ** 2
        self._velocity += self._config.frequency_ratio * acceleration
    
    def _is_agent_outside_boundary(self):
        return np.any(self._agent_position < 0) \
               or np.any(self._agent_position > np.asarray([self._width, self._config.length]))
    
    def _is_agent_outside_shaping_boundary(self):
        return np.any(self._agent_position < self._config.reward_shaping_slack) \
               or np.any(self._agent_position > np.asarray([self._width, self._config.length]) - self._config.reward_shaping_slack)

    def _update_time(self):
        # Assume spaceship frequency to be one
        self._current_time += self._config.frequency_ratio
    
    def _get_noisy_target_position(self):
        return self._target_position + \
               np.random.normal(0, self._config.target_noise_std, 2)

    def get_num_constraints(self):
        return 4

    def get_constraint_values(self):
        # There a a total of 4 constraints
        # a lower and upper bound for each dim
        # We define all the constraints such that C_i = 0
        # _agent_position > 0 + _agent_slack => -_agent_position + _agent_slack < 0
        min_constraints = self._config.agent_slack - self._agent_position
        # _agent_position < np.asarray([_width, length]) - agent_slack
        # => _agent_position + agent_slack - np.asarray([_width, length]) < 0
        max_constraint = self._agent_position  + self._config.agent_slack \
                         - np.asarray([self._width, self._config.length])

        return np.concatenate([min_constraints, max_constraint])

    def step(self, action):
        # Increment time
        self._update_time()

        # Calculate new position of the agent
        self._move_agent(action)

        # Find reward         
        reward = self._get_reward()
        
        # Prepare return payload
        observation = {
            "agent_position": self._agent_position,
            "agent_velocity": self._velocity,
            "target_position": self._get_noisy_target_position()
        }
        flat_observation = flatten(self.dict_observation_space, observation)

        terminated = self._is_agent_outside_boundary()
        truncated = int(self._current_time // 1) >= self._episode_length
        
        return flat_observation, reward, terminated, truncated, self.get_info()