# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Safety-Gymnasium Environments."""

import copy
# import sys
# sys.path.insert(0,'/path/to/mod_directory')

from gymnasium import make as gymnasium_make
from gymnasium import register as gymnasium_register

from . import vector, wrappers
from .utils.registration import make, register
from .version import __version__

#-----------------------------
# allow for rendering on headless servers
import os
bashrc_path = os.path.expanduser('~/.bashrc')

# Write the export command to .bashrc
with open(bashrc_path, 'a') as bashrc_file:
    bashrc_file.write("export MUJOCO_GL=egl\n")

# Reload the environment to apply changes
os.system("source ~/.bashrc")
#----------------------------

__all__ = [
    'register',
    'make',
    'gymnasium_make',
    'gymnasium_register',
]

VERSION = 'v0'
ROBOT_NAMES = ('Point', 'Car', 'Doggo', 'Racecar', 'Ant')
MAKE_VISION_ENVIRONMENTS = True
MAKE_DEBUG_ENVIRONMENTS = True

# ========================================
# Helper Methods for Easy Registration
# ========================================

PREFIX = 'Safety'

robots = ROBOT_NAMES


def __register_helper(env_id, entry_point, spec_kwargs=None, **kwargs):
    """Register a environment to both Safety-Gymnasium and Gymnasium registry."""
    env_name, dash, version = env_id.partition('-')
    if spec_kwargs is None:
        spec_kwargs = {}

    register(
        id=env_id,
        entry_point=entry_point,
        kwargs=spec_kwargs,
        **kwargs,
    )
    gymnasium_register(
        id=f'{env_name}Gymnasium{dash}{version}',
        entry_point='continuousSafetyGym.safety_gymnasium.wrappers.gymnasium_conversion:make_gymnasium_environment',
        kwargs={'env_id': f'{env_name}Gymnasium{dash}{version}', **copy.deepcopy(spec_kwargs)},
        **kwargs,
    )


def __combine(tasks, agents, max_episode_steps):
    """Combine tasks and agents together to register environment tasks."""
    for task_name, task_config in tasks.items():
        # Vector inputs
        for robot_name in agents:
            env_id = f'{PREFIX}{robot_name}{task_name}-{VERSION}'
            combined_config = copy.deepcopy(task_config)
            combined_config.update({'agent_name': robot_name})

            __register_helper(
                env_id=env_id,
                entry_point='continuousSafetyGym.safety_gymnasium.builder:Builder',
                spec_kwargs={'config': combined_config, 'task_id': env_id},
                max_episode_steps=max_episode_steps,
            )

            if MAKE_VISION_ENVIRONMENTS:
                # Vision inputs
                vision_env_name = f'{PREFIX}{robot_name}{task_name}Vision-{VERSION}'
                vision_config = {
                    'observe_vision': True,
                    'observation_flatten': False,
                }
                vision_config.update(combined_config)
                __register_helper(
                    env_id=vision_env_name,
                    entry_point='continuousSafetyGym.safety_gymnasium.builder:Builder',
                    spec_kwargs={'config': vision_config, 'task_id': env_id},
                    max_episode_steps=max_episode_steps,
                )

            if MAKE_DEBUG_ENVIRONMENTS and robot_name in ['Point', 'Car', 'Racecar']:
                # Keyboard inputs for debugging
                debug_env_name = f'{PREFIX}{robot_name}{task_name}Debug-{VERSION}'
                debug_config = {'debug': True}
                debug_config.update(combined_config)
                __register_helper(
                    env_id=debug_env_name,
                    entry_point='continuousSafetyGym.safety_gymnasium.builder:Builder',
                    spec_kwargs={'config': debug_config, 'task_id': env_id},
                    max_episode_steps=max_episode_steps,
                )


# ----------------------------------------
# Safety Navigation
# ----------------------------------------

# Button Environments
# ----------------------------------------
button_tasks = {
    'ButtonBase': {'task_name': 'ButtonBase'},
    'Button0': {},
    'Button1': {},
    'Button2': {},
}
__combine(button_tasks, robots, max_episode_steps=1000)


# Push Environments
# ----------------------------------------
push_tasks = {'PushBase': {'task_name': 'PushBase'}, 'Push0': {}, 'Push1': {}, 'Push2': {}}
__combine(push_tasks, robots, max_episode_steps=1000)


# Goal Environments
# ----------------------------------------
goal_tasks = {'GoalBase': {'task_name': 'GoalBase'}, 'Goal0': {}, 'Goal1': {}, 'Goal2': {}}
__combine(goal_tasks, robots, max_episode_steps=1000)


# Circle Environments
# ----------------------------------------
circle_tasks = {
    'CircleBase': {'task_name': 'CircleBase'},
    'Circle0': {},
    'Circle1': {},
    'Circle2': {},
}
__combine(circle_tasks, robots, max_episode_steps=500)


# Run Environments
# ----------------------------------------
run_tasks = {'RunBase': {'task_name': 'RunBase'}, 'Run0': {}}
__combine(run_tasks, robots, max_episode_steps=500)

