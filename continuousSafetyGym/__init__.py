from gymnasium.envs.registration import register

# Multiagent envs
# ----------------------------------------

register(
    id='MultiagentDescentralizedSafe-v0',
    entry_point='continuousSafetyGym.multiagent_particle.decentralized_safe:DecentralizedSafe',
    kwargs=dict(
        safe_init=True,
    ),
    # FIXME(cathywu) currently has to be exactly max_path_length parameters in
    # rllab run script
    max_episode_steps=300,
)


# ===== Safe Explorer =====
register(
    id='SpaceshipSafe-v0',
    entry_point='continuousSafetyGym.safe_explorer.spaceship:Spaceship',
    # FIXME(cathywu) currently has to be exactly max_path_length parameters in
    # rllab run script
    max_episode_steps=300,
)


# ===== Bullet Safety Gym =====
register(
    id='ContSafetyBallReach-v0',
    entry_point='continuousSafetyGym.bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=500,
    kwargs=dict(
        agent='Ball',
        task='ReachGoalTask',
        obstacles={'Pillar': {'number': 8, 'fixed_base': True,
                              'movement': 'static'}
                   },
        world={'name': 'SmallRoom'},
        # debug=True
    ),
)


register(
    id='ContSafetyBallReach-v1',
    entry_point='continuousSafetyGym.bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=500,
    kwargs=dict(
        agent='Ball',
        task='ReachGoalTask',
        obstacles={'Box': {'number': 3, 'fixed_base': False,
                           'movement': 'circular'},
                   'Pillar': {'number': 5, 'fixed_base': True,
                              'movement': 'static'}
                   },
        world={'name': 'SmallRoom'},
    ),
)


register(
    id='ContSafetyBallGather-v0',
    entry_point='continuousSafetyGym.bullet_safety_gym.envs.builder:EnvironmentBuilder',
    max_episode_steps=500,
    kwargs=dict(
        agent='Ball',
        task='GatherTask',
        obstacles={'Apple': {'number': 8, 'fixed_base': True,
                           'movement': 'static'},
                   'Bomb': {'number': 8, 'fixed_base': True,
                              'movement': 'static'}
                   },
        world={'name': 'SmallRoom', 'factor': 1},
        # debug=True
    ),
)
